# ---
# predicting the time evolution of the stellar bispectrum
# ---

import numpy as np
import jax
import jax.numpy as jnp

from .training_routines import TrainerModule


# make derrived training class
class StellarBispectrumTrainer(TrainerModule):
    def batch_to_input(self, batch):
        inp_data, _, _ = batch
        return inp_data

    def get_loss_function(self):
        # Function for calculating loss and accuracy for a batch
        def calculate_loss(params, rng, batch, train):
            inp_data, _, labels = batch
            rng, dropout_apply_rng = random.split(rng)
            logits = self.model.apply(
                {"params": params},
                inp_data,
                add_positional_encoding=False,  # No positional encoding since this is a permutation equivariant task
                train=train,
                rngs={"dropout": dropout_apply_rng},
            )
            logits = logits.squeeze(axis=-1)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, labels
            ).mean()
            acc = (logits.argmax(axis=-1) == labels).astype(jnp.float32).mean()
            return loss, (acc, rng)

        return calculate_loss
