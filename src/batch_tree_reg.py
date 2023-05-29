### batched regularizes a model towards producing certain independences
import torch
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
import random 
import argparse

from transformers import GPT2Config, AutoTokenizer
# TODO: change this to inheritance
from .TreeRegularizer.GPT2_model import GPT2Model
from .TreeRegularizer.tree_projection_src import batched_tree_projection
from .data_utils import build_datasets_babylm

device = torch.device("cuda")

class Regularizer():
    def __init__(self, model, sim_metric, tokenizer, start_relax_layer):
        self.sim_metric = sim_metric
        self.tokenizer = tokenizer
        self.start_relax_layer = start_relax_layer
        self.model = model
        self.tree_projector = batched_tree_projection.TreeProjection(self.model, self.tokenizer, sim_fn=self.sim_metric, normalize=True)

    def compute_vectors(self, input_strs):
        '''
            Takes a model, input_str (batch_size of string of tokens, possibly tokenized by GPT2) and returns:
            1. Contextual vectors: (batch_size x num_words x hidden_dim)
            2. Context Free vectors: (batch_size x num_spans x hidden_dim)
            3. span2idx: batches of span-index correlation fo each of the samples. For each span, the index of the span in the context free vectors
        '''
        sent_tokens, idxs = batched_tree_projection.get_pre_tokenized_info(self.tokenizer, input_strs)
        sentence2idx_tuple, masked_strs, input_masks = self.tree_projector._get_masking_info(input_strs)
        
        outer_context_vecs = batched_tree_projection.get_all_hidden_states(
            self.model,
            self.tokenizer,
            input_strs,
            tqdm_disable=True,
            pre_tokenized=(sent_tokens, idxs),
            regularizer=True
        )

        inner_context_vecs = batched_tree_projection.get_all_hidden_states(
            self.model,
            self.tokenizer,
            masked_strs,
            input_masks,
            sum_all=True,
            start_relax_layer=self.start_relax_layer,
            tqdm_disable=True,
            ## MOD: not sure why we need pre_tokenized here since we return after sum_all
            #pre_tokenized=(sent_tokens * len(masked_strs), idxs * len(masked_strs)),
            regularizer=True
        )

        span2idx = []
        for keys in sentence2idx_tuple:
            cur_span2idx = {}
            for idx, key in keys:
                st, en = key
                cur_span2idx[(st, en)] = idx
            span2idx.append(cur_span2idx)
        
        return outer_context_vecs, inner_context_vecs, span2idx


    def __call__(self, input_strs):
        contextual_vectors, context_free_vectors, total_span2idx = self.compute_vectors(input_strs)
        ## TODO: figure out how to make get_score into batches
        #print(total_span2idx)
        total_score = 0.0
        for i in range(len(input_strs)):
            input_str = input_strs[i]
            span2idx = total_span2idx[i]
            num_words = len(input_str.split(' '))
            cur_score = self.get_score(0, num_words-1, contextual_vectors[i][-1], context_free_vectors, span2idx)
            #print("cur_score:", cur_score)
            total_score += cur_score
        return total_score/len(input_strs)

    def get_score(self, i, j, contextual_vectors, context_free_vectors, span2idx):
        if (i == j):
            return 0.0
        all_scores = {}
        for k in range(i, j):
            i1 = span2idx[(i, k)]
            i2 = span2idx[(k+1, j)]
            cont_vec_1 = contextual_vectors[i: k+1].sum(axis=0)
            inner_vec_1 = context_free_vectors[i1][-1]

            cont_vec_2 = contextual_vectors[k+1:j+1].sum(axis=0)
            inner_vec_2 = context_free_vectors[i2][-1]
            all_scores[k] = self.sim_metric(cont_vec_1, inner_vec_1) + self.sim_metric(cont_vec_2, inner_vec_2)

        best_split = max(all_scores, key=all_scores.get)
        random_split = random.choice(list(all_scores.keys()))

        score_best = all_scores[best_split]
        score_random = all_scores[random_split]

        s1 = self.get_score(i, best_split, contextual_vectors, context_free_vectors, span2idx)
        s2 = self.get_score(best_split+1, j, contextual_vectors, context_free_vectors, span2idx)
        return self.loss_type(score_best, score_random) + s1 + s2

    def loss_type(self, best_score, random_score):
        return -1.0*(best_score - random_score)


def load_model(model_name):
    config = GPT2Config()
    model = GPT2Model.from_pretrained(
        "exp/" + model_name + "/" + model_name + "-run-42", config = config
    ) 
    tokenizer = AutoTokenizer.from_pretrained(
        'gpt2'
    )
    return model, tokenizer

if __name__ == '__main__':
    ### Load Model
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="debug-gpt2-small-babylm_10M")
    parser.add_argument('--dataset', type=str, default='babylm')
    # NOTE: when batch_size = 8, out of memory in 3 batches
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--output_path', type=str, default='exp/reg-debug-gpt2-small-babylm_10M')
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_name)
    model.to(device)
    print("Model and tokenizer loaded")
    reg = Regularizer(model = model, sim_metric=torch.nn.CosineSimilarity(dim=0), tokenizer=tokenizer, start_relax_layer=0)
    print("Regularizer initiated")
   
    if args.dataset == "babylm":
        input_strings = build_datasets_babylm()
    print("Input_strings loaded")
    ### create optimizer
    optimizer = Adam(model.parameters(), lr=1e-4)
    
    with tqdm(total=len(input_strings)) as pbar:
        for i in range(0, len(input_strings), args.batch_size):
            #end = min(i + args.batch_size, len(input_strings))
            #input_strs = ["I love red apples.", "I love red abcadabras."]
            input_strs = input_strings[i: i+ args.batch_size]
            loss = reg(input_strs)
            print("-"*20)
            print("LOSS:", loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(args.batch_size)

    ### make this batched
    ### add this into the training loop
    ### overall_loss: -log P(x_i | x_<i) 
    ### if regularizer:
    ###     overall_loss += self.reg(batch_of_strs)
    