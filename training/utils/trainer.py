"""The HF Trainer both training codes subclass."""

import torch
from transformers import Trainer
from transformers.trainer_utils import SaveStrategy


class CustomLoggingTrainer(Trainer):
    """A Trainer whose training logs also carry the values passed to `custom_log`, averaged over the logging interval.

    Subclasses pass themselves as a callback (`callbacks=[self]`), so `on_train_begin` runs.
    """

    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        model_init=None,
        compute_loss_func=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        optimizer_cls_and_kwargs=None,
        preprocess_logits_for_metrics=None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self._custom_metrics = {}

    def on_train_begin(self, args, state, control, **kwargs):
        # a resume restores the logging / eval / save intervals from trainer_state.json; the arguments' values apply
        state.logging_steps = args.logging_steps
        state.eval_steps = args.eval_steps
        state.save_steps = args.save_steps

    def custom_log(self, log_dict: dict[str, torch.Tensor | float]):
        """Accumulate values (per microbatch, divided by the accumulation steps) for the next training log."""
        for key, value in log_dict.items():
            if key not in self._custom_metrics:
                self._custom_metrics[key] = 0.0
            if torch.is_tensor(value):
                self._custom_metrics[key] += value.mean().item() / self.current_gradient_accumulation_steps
            else:
                self._custom_metrics[key] += value / self.current_gradient_accumulation_steps

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: dict[str, float] = {}

            # the loss averaged over all processes
            tr_loss_scalar = self.accelerator.gather(tr_loss).mean().item()

            tr_loss -= tr_loss

            steps = self.state.global_step - self._globalstep_last_logged
            logs["loss"] = round(tr_loss_scalar / steps, 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            if self._custom_metrics is not None:
                logs.update({k: v / steps for k, v in self._custom_metrics.items()})
                self._custom_metrics.clear()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
