import argparse
import copy

import numpy as np
import torch

from utils import (
    get_dataset,
    get_model,
    get_unadapt_method,
    load_config,
    seed_all,
    test,
    train,
)


def calculate_loss_gap_ratio(losses_unadapt, losses_base):
    loss_max = losses_base[
        0
    ]  # Maximum loss, since you could always use the base model for the fine-tune task zero-shot
    # Min the losses_base with loss_max so it doesn't overcount from getting higher loss than the base_pt model
    losses_unadapt_clamped = np.minimum(
        loss_max, np.array(losses_unadapt, dtype=np.float32)
    )
    # Gap between the "unadaptable" model and the base model
    loss_gap_ulm_alm = np.trapz(
        losses_unadapt_clamped - np.array(losses_base, dtype=np.float32)
    )
    # Gap between the base model and the maximum loss
    loss_gap_max_alm = np.trapz(loss_max - np.array(losses_base, dtype=np.float32))
    # Ratio of the two gaps -- 100% means the unadaptable model is as bad as if you didn't do any fine-tuning
    loss_gap_ratio = loss_gap_ulm_alm / loss_gap_max_alm
    return loss_gap_ratio


def calculate_unadaptability_metrics(
    model,
    unadapt_method,
    device,
    config,
    pre_train_loader,
    pre_test_loader,
    fine_train_loader,
    fine_test_loader,
):
    # Make the model unadaptable; optionally use pretraining dataset
    ufm_model = copy.deepcopy(model)
    unadapt_method(ufm_model, pre_train_loader)

    # Calculate relative accuracy on pretraining test dataset
    _, ufm_pre_acc = test(ufm_model, device, pre_test_loader)

    # Finetune unadaptable model on finetuning dataset
    ufm_fine_losses = train(
        ufm_model,
        device,
        fine_train_loader,
        config.finetune.epochs,
        learning_rate=config.finetune.lr,
    )
    _, ufm_fine_acc = test(ufm_model, device, fine_test_loader)

    return ufm_pre_acc, ufm_fine_acc, ufm_fine_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mnist.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    seed_all(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = get_model(config.model, device)
    pre_train_loader, pre_test_loader = get_dataset(
        config.pretrain.dataset,
        config.pretrain.batch_size,
        config.pretrain.test_batch_size,
    )
    fine_train_loader, fine_test_loader = get_dataset(
        config.finetune.dataset,
        config.finetune.batch_size,
        config.finetune.test_batch_size,
    )

    if not config.pretrained:
        train(
            model,
            device,
            pre_train_loader,
            config.pretrain.epochs,
            learning_rate=config.pretrain.lr,
        )

    _, pre_acc = test(model, device, pre_test_loader)

    # Finetune model
    fine_model = copy.deepcopy(model)
    model_fine_losses = train(
        fine_model,
        device,
        fine_train_loader,
        config.finetune.epochs,
        learning_rate=config.finetune.lr,
    )
    _, model_fine_acc = test(fine_model, device, fine_test_loader)

    for method_config in config.unadapt:
        unadapt_method = get_unadapt_method(method_config)
        ufm_pre_acc, ufm_fine_acc, ufm_fine_losses = calculate_unadaptability_metrics(
            model,
            unadapt_method,
            device,
            config,
            pre_train_loader,
            pre_test_loader,
            fine_train_loader,
            fine_test_loader,
        )
        pre_acc_ratio = ufm_pre_acc / pre_acc
        fine_acc_ratio = ufm_fine_acc / model_fine_acc
        loss_gap_ratio = calculate_loss_gap_ratio(ufm_fine_losses, model_fine_losses)

        print(f"Unadaptable method: {method_config}")
        print(f"Model acc: {pre_acc}")
        print(f"UFM acc: {ufm_pre_acc}")
        print(f"Pretrained acc ratio: {pre_acc_ratio: .4f}")
        print(f"Fine-tuned model acc: {model_fine_acc}")
        print(f"Fine-tuned UFM acc: {ufm_fine_acc}")
        print(f"Fine-tuned acc ratio: {fine_acc_ratio: .4f}")
        print(f"Loss gap ratio: {loss_gap_ratio: .4f}")


if __name__ == "__main__":
    main()
