import os
import argparse

from dataset.dataloader import load_data, get_loader
from dataset.field import Vocab
from utils import seq2sen

from transformer import make_model, Batch, get_std_opt,\
                        SimpleLossCompute, LabelSmoothing
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import datetime

def main(args):
    src, tgt = load_data(args.path)

    src_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    src_vocab.load(os.path.join(args.path, 'vocab.en'))
    tgt_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    tgt_vocab.load(os.path.join(args.path, 'vocab.de'))

    sos_idx = 0
    eos_idx = 1
    pad_idx = 2
    max_length = 50

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    # Set hyper parameter
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = make_model(src_vocab_size, tgt_vocab_size).to(device)
    optimizer = get_std_opt(model)
    criterion = LabelSmoothing(size=tgt_vocab_size, padding_idx = pad_idx, smoothing=0.1)
    train_criterion = SimpleLossCompute(model.generator, criterion, optimizer)
    valid_criterion = SimpleLossCompute(model.generator, criterion, None)
    print('Using device:', device)

    if not args.test:
        train_loader = get_loader(src['train'], tgt['train'], src_vocab, tgt_vocab, batch_size=args.batch_size, shuffle=True)
        valid_loader = get_loader(src['valid'], tgt['valid'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        best_loss = float('inf')
        for epoch in range(args.epochs):
            train_total_loss, valid_total_loss = 0.0, 0.0
            start = time.time()
            total_tokens = 0
            tokens = 0

            model.train()
            # Train
            for src_batch, tgt_batch in train_loader:
                src_batch = torch.tensor(src_batch).to(device)
                tgt_batch = torch.tensor(tgt_batch).to(device)
                batch = Batch(src_batch, tgt_batch, pad_idx)

                prediction = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
                loss = train_criterion(prediction, batch.trg_y, batch.ntokens)

                train_total_loss += loss
                total_tokens += batch.ntokens
                tokens += batch.ntokens

            # Valid
            model.eval()
            for src_batch, tgt_batch in valid_loader:
                src_batch = torch.tensor(src_batch).to(device)
                tgt_batch = torch.tensor(tgt_batch).to(device)
                batch = Batch(src_batch, tgt_batch, pad_idx)

                prediction = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
                loss = valid_criterion(prediction, batch.trg_y, batch.ntokens)

                valid_total_loss += loss
                total_tokens += batch.ntokens
                tokens += batch.ntokens

            if valid_total_loss < best_loss:
                best_loss = valid_total_loss
                best_model_state = model.state_dict()
                best_optimizer_state = optimizer.optimizer.state_dict()

            elpsed = time.time() - start
            print(f"""{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} || [{epoch}/{args.epochs}], train_loss = {train_total_loss:.4f}, valid_loss = {valid_total_loss:.4f}, Tokens per Sec = {tokens / elpsed}""")
            tokens = 0
            start = time.time()
            
        # Save model
        torch.save({
            'epoch': args.epochs,
            'model_state_dict': best_model_state,
            'optimizer_state': best_optimizer_state,
            'loss': best_loss
        }, f"{args.model_dir}/best.pt")
        print("Model saved")
    else:
        # Load the model
        checkpoint = torch.load(f"{args.model_dir}/best.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state'])
        model.eval()
        print("Model loaded")
    
        # Test
        test_loader = get_loader(src['test'], tgt['test'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        pred = []

        for src_batch, tgt_batch in test_loader:
            src_batch = torch.tensor(src_batch).to(device)
            tgt_batch = torch.tensor(tgt_batch).to(device)
            batch = Batch(src_batch, tgt_batch, pad_idx)
            
            # Get pred_batch
            memory = model.encode(batch.src, batch.src_mask)
            pred_batch = torch.ones(src_batch.size(0), 1)\
                            .fill_(sos_idx).type_as(batch.src.data).to(device)
            for i in range(max_length-1):
                out = model.decode(memory, batch.src_mask,
                                    Variable(pred_batch),
                                    Variable(Batch.make_std_mask(pred_batch, pad_idx)
                                            .type_as(batch.src.data)))
                prob = model.generator(out[:, -1])
                prob.index_fill_(1, torch.tensor([sos_idx, pad_idx]).to(device), -float('inf'))
                _, next_word = torch.max(prob, dim = 1)

                    
                pred_batch = torch.cat([pred_batch, next_word.unsqueeze(-1)], dim=1)
            pred_batch = torch.cat([pred_batch, torch.ones(src_batch.size(0), 1)\
                                                    .fill_(eos_idx).type_as(batch.src.data).to(device)], dim=1)

            # every sentences in pred_batch should start with <sos> token (index: 0) and end with <eos> token (index: 1).
            # every <pad> token (index: 2) should be located after <eos> token (index: 1).
            # example of pred_batch:
            # [[0, 5, 6, 7, 1],
            #  [0, 4, 9, 1, 2],
            #  [0, 6, 1, 2, 2]]
            pred += seq2sen(pred_batch.tolist(), tgt_vocab)

        with open('results/pred.txt', 'w') as f:
            for line in pred:
                f.write('{}\n'.format(line))

        os.system('bash scripts/bleu.sh results/pred.txt multi30k/test.de.atok')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument(
        '--path',
        type=str,
        default='multi30k')

    parser.add_argument(
        '--epochs',
        type=int,
        default=100)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128)

    parser.add_argument(
        '--test',
        action='store_true')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./model'
    )
    args = parser.parse_args()

    main(args)
