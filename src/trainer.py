from torch import nn
from transformers import Trainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

class TreeTrainer(Trainer):
    def __init__(self, regularizer, max_batch_size = 2, **kwargs,):
        super().__init__(**kwargs)
        self.regularizer = regularizer
        self.max_batch_size = max_batch_size

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # Calculate custom tree loss
        t_loss = self.tree_loss(model, inputs)

        # Compute default loss function    
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Combine tree loss with default loss
        combined_loss = loss + t_loss
        print("combined_loss:", combined_loss)
        return (combined_loss, outputs) if return_outputs else combined_loss

    def tree_loss(self, model, input):
        """
        This function calculates the tree regularization loss of the given model on the input strings.
        """
        # Convert inputs to strings
        # TODO: clean up input strings
        input_strs = [self.tokenizer.decode(string) for string in input["input_ids"]]
        input_strs = [sentence for text in input_strs for sentence in text.split('\n')]
        print(len(input_strs))
        loss = 0.0
        # Use smaller batch sizes than the input size so there won't be memory issues
        for i in range(0, len(input_strs), self.max_batch_size):
            end = min(i + self.max_batch_size, len(input_strs))
            input_str = input_strs[i: end]
            print(input_str)
            loss += self.regularizer(input_str)
            print("tree loss:",loss)
        return loss