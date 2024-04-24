"""Metrics for evaluating unadaptability and relative pre-training performance."""


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
    unadapt_config,
    device,
    config,
    pre_train_loader,
    pre_test_loader,
    fine_train_loader,
    fine_test_loader,
):
    # Make the model unadaptable; optionally use pretraining dataset
    ufm_model = copy.deepcopy(model)
    ufm_model = get_unadaptable_model(
        ufm_model, unadapt_config, device, pre_train_loader
    )

    # Calculate relative accuracy on pretraining test dataset
    _, ufm_pre_acc = test(ufm_model, device, pre_test_loader)

    # Finetune unadaptable model on finetuning dataset
    # TODO -- Move to fine_tuning module
    # ufm_fine_losses = train(
    #     ufm_model,
    #     device,
    #     fine_train_loader,
    #     config.finetune.epochs,
    #     learning_rate=config.finetune.lr,
    # )
    # _, ufm_fine_acc = test(ufm_model, device, fine_test_loader)

    return ufm_pre_acc, ufm_fine_acc, ufm_fine_losses
