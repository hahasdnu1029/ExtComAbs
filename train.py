#!/usr/bin/python
# -*- coding: utf-8 -*-
# __author__="Pengyu Yang"

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import argparse
import datetime
import os
import shutil
import time

import torch
from model import GengrateDocument, Classfiler
from tester import SLTester
from module.dataloader import ExampleSet
from rouge import Rouge
from module.embedding import Word_Embedding
from module.vocabulary import Vocab
from tools.logger import *


def save_model(model, save_file):
    """save model use torch.save()
    :param model:
    :param save_file:
    :return:
    """
    with open(save_file, 'wb') as f:
        torch.save(model.state_dict(), f)
    logger.info('[INFO] Saving model to %s', save_file)


def setup_training(model, train_loader, valid_loader, valset, args ,vocab):
    """ Does setup before starting training (run_training)
    
        :param model: the model
        :param train_loader: train dataset loader
        :param valid_loader: valid dataset loader
        :param valset: valid dataset which includes text and summary
        :param args: args for model
        :return:
    """

    train_dir = os.path.join(args.save_root, "train_gen")
    if os.path.exists(train_dir) and args.restore_model != 'None':
        logger.info("[INFO] Restoring %s for training...", args.restore_model)
        bestmodel_file = os.path.join(train_dir, args.restore_model)
        model.load_state_dict(torch.load(bestmodel_file))
        args.save_root = args.save_root + "_reload"
    else:
        logger.info("[INFO] Create new model for training...")
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.makedirs(train_dir)

    try:
        run_training(model, train_loader, valid_loader, valset, args, train_dir, vocab)
    except KeyboardInterrupt:
        logger.error("[Error] Caught keyboard interrupt on worker. Stopping supervisor...")
        save_model(model, os.path.join(train_dir, "earlystop"))


def run_training(model, train_loader, valid_loader, valset, args, train_dir, vocab):
    """  Repeatedly runs training iterations, logging loss to screen and log files

            :param model: the model
            :param train_loader: train dataset loader
            :param valid_loader: valid dataset loader
            :param valset: valid dataset which includes text and summary
            :param args: args for model
            :param train_dir: where to save checkpoints
            :return:
        """
    logger.info("[INFO] Starting run_training_gen")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    criterion = torch.nn.CrossEntropyLoss()

    best_train_loss = None

    for epoch in range(1, args.n_epochs + 1):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        train_loss = 0.0
        for i, (features_in, features_out, labels, indexs) in enumerate(train_loader):

            model.train()

            iter_start_time = time.time()

            if args.cuda:
                features_in = features_in.to(torch.device("cuda"))
                features_out = features_out.to(torch.device("cuda"))

            outputs = model(features_in, features_out)

            loss = None
            for j in range(len(outputs)):
                feature_out = features_out[j].repeat(outputs[j].shape[0], 1)
                output = outputs[j].transpose(2, 1)
                if j == 0:
                    loss = criterion(output, feature_out)
                else:
                    loss.add_(criterion(output, feature_out))

            loss = loss / args.batch_size

            optimizer.zero_grad()

            loss.backward()

            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()

            train_loss += float(loss.data)
            epoch_loss += float(loss.data)

            if i !=0 and i % 100 == 0:
                logger.info('       | end of iter {:3d} | time: {:5.2f}s | train loss {:5.4f} | '
                            .format(i, (time.time() - iter_start_time), float(train_loss/100)))
                train_loss = 0.0

        # 随epoch增大，学习虑逐渐降低
        if args.lr_descent:
            new_lr = max(5e-6, args.lr / (epoch + 1))
            for param_group in list(optimizer.param_groups):
                param_group['lr'] = new_lr
            logger.info("[INFO] The learning rate now is %f", new_lr)

        # 每个epoch的平均loss
        epoch_avg_loss = epoch_loss / len(train_loader)
        logger.info('   | end of epoch {:3d} | time: {:5.2f}s | epoch train loss {:5.4f} | '
                    .format(epoch, (time.time() - epoch_start_time), float(epoch_avg_loss)))

        # epoch_avg_loss<best_train_loss save新的model，并吧best_train_loss=epoch_avg_loss
        # epoch_avg_loss>=best_train_loss earlystop（loss未下降，停止训练）
        if not best_train_loss or epoch_avg_loss < best_train_loss:
            save_file = os.path.join(train_dir, "bestmodel")
            logger.info('[INFO] Found new best model with %.3f running_train_loss. Saving to %s', float(epoch_avg_loss),
                        save_file)
            save_model(model, save_file)
            best_train_loss = epoch_avg_loss
        elif epoch_avg_loss >= best_train_loss:
            logger.error("[Error] training loss does not descent. Stopping supervisor...")
            save_model(model, os.path.join(train_dir, "earlystop"))
            sys.exit(1)


def run_training_class(model_gen, model_class, train_loader, valid_loader, valset, args, vocab):
    """  Repeatedly runs training iterations, logging loss to screen and log files

            :param model_gen: the model gen doc
            :param model_class: the model classfiler
            :param train_loader: train dataset loader
            :param valid_loader: valid dataset loader
            :param valset: valid dataset which includes text and summary
            :param args: args for model
            :param train_dir: where to save checkpoints
            :return:
        """
    logger.info("[INFO] Starting run_training_class")

    train_gen_dir = os.path.join(args.save_root, "train_gen")
    bestmodel_file = os.path.join(train_gen_dir, "bestmodel")
    model_gen.load_state_dict(torch.load(bestmodel_file))

    train_class_dir = os.path.join(args.save_root, "train_class")

    logger.info("[INFO] Create new model for training...")
    if os.path.exists(train_class_dir):
        shutil.rmtree(train_class_dir)
    os.makedirs(train_class_dir)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_class.parameters()), lr=args.lr)

    criterion = torch.nn.CrossEntropyLoss()

    best_train_loss = None
    best_loss = None
    best_f = None
    non_descent_cnt = 0
    save_no = 0

    for epoch in range(1, args.n_epochs + 1):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        train_loss = 0.0
        for i, (features_in, features_out, labels, indexs) in enumerate(train_loader):

            model_class.train()

            iter_start_time = time.time()

            if args.cuda:
                features_in = features_in.to(torch.device("cuda"))
                features_out = features_out.to(torch.device("cuda"))
                labels = labels.to(torch.device("cuda"))

            outputs = model_class(model_gen, features_in, features_out)
            outputs = torch.transpose(outputs, 2, 1)
            print(outputs.data)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model_class.parameters(), args.max_grad_norm)

            optimizer.step()

            train_loss += float(loss.data)
            epoch_loss += float(loss.data)

            if i != 0 and i % 100 == 0:
                logger.info('       | end of iter {:3d} | time: {:5.2f}s | train loss {:5.4f} | '
                            .format(i, (time.time() - iter_start_time), float(train_loss/100)))
                train_loss = 0.0

        # 随epoch增大，学习虑逐渐降低
        if args.lr_descent:
            new_lr = max(5e-6, args.lr / (epoch + 1))
            for param_group in list(optimizer.param_groups):
                param_group['lr'] = new_lr
            logger.info("[INFO] The learning rate now is %f", new_lr)

        # 每个epoch的平均loss
        epoch_avg_loss = epoch_loss / len(train_loader)
        logger.info('   | end of epoch {:3d} | time: {:5.2f}s | epoch train loss {:5.4f} | '
                    .format(epoch, (time.time() - epoch_start_time), float(epoch_avg_loss)))

        # epoch_avg_loss<best_train_loss save新的model，并吧best_train_loss=epoch_avg_loss
        # epoch_avg_loss>=best_train_loss earlystop（loss未下降，停止训练）
        if not best_train_loss or epoch_avg_loss < best_train_loss:
            save_file = os.path.join(train_class_dir, "bestmodel")
            logger.info('[INFO] Found new best model with %.3f running_train_loss. Saving to %s', float(epoch_avg_loss),
                        save_file)
            save_model(model_class, save_file)
            best_train_loss = epoch_avg_loss
        elif epoch_avg_loss >= best_train_loss:
            logger.error("[Error] training loss does not descent. Stopping supervisor...")
            save_model(model_class, os.path.join(train_class_dir, "earlystop"))
            sys.exit(1)

        # 每个epoch结束对模型进行评估，best_loss,best_F,non_descent_cnt,saveNo
        now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        best_loss, best_f, non_descent_cnt, save_no = run_eval(model_gen, model_class, valid_loader, valset, args, best_loss, best_f, non_descent_cnt, save_no, vocab, now_time)

        # model评估连续3次未下降停止训练
        if non_descent_cnt >= 3:
            logger.error("[Error] val loss does not descent for three times. Stopping supervisor...")
            save_model(model_class, os.path.join(train_class_dir, "earlystop"))
            return


def run_eval(model_gen, model_class, loader, valset, args, best_loss, best_f, non_descent_cnt, save_no, vocab, now_time):
    ''' 
        Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far.
        :param model_gen: the model gen doc
        :param model_class: the model classfiler
        :param loader: valid dataset loader
        :param valset: valid dataset which includes text and summary
        :param args: hps for model
        :param best_loss: best valid loss so far
        :param best_F: best valid F so far
        :param non_descent_cnt: the number of non descent epoch (for early stop)
        :param saveNo: the number of saved models (always keep best saveNo checkpoints)
        :return: 
    '''
    logger.info("[INFO] Starting eval for this model ...")
    eval_dir = os.path.join(args.save_root, "eval")  # make a subdir of the root dir for eval data
    if not os.path.exists(eval_dir): os.makedirs(eval_dir)

    model_class.eval()

    iter_start_time = time.time()

    with torch.no_grad():
        tester = SLTester(model_gen, model_class, args.m)
        for i, (features_in, features_out, labels, indexs) in enumerate(loader):
            if(args.cuda):
                features_in = features_in.to(torch.device("cuda"))
                features_out = features_out.to(torch.device("cuda"))
                labels = labels.to(torch.device("cuda"))
            tester.evaluation(features_in, features_out, labels, indexs, valset, now_time, vocab, True)

    running_avg_loss = tester.running_avg_loss

    print(tester.running_loss)
    print(running_avg_loss)

    if len(tester.hyps) == 0 or len(tester.refer) == 0:
        logger.error("During testing, no hyps is selected!")
        return
    rouge = Rouge()
    scores_all = rouge.get_scores(tester.hyps, tester.refer, avg=True)
    logger.info('[INFO] End of valid | time: {:5.2f}s | valid loss {:5.4f} | ' .format((time.time() - iter_start_time), float(running_avg_loss)))

    res = "Rouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-1']['p'], scores_all['rouge-1']['r'], scores_all['rouge-1']['f']) \
          + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-2']['p'], scores_all['rouge-2']['r'], scores_all['rouge-2']['f']) \
          + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-l']['p'], scores_all['rouge-l']['r'], scores_all['rouge-l']['f'])

    logger.info(res)

    tester.getMetric()

    F = tester.labelMetric

    if best_loss is None or running_avg_loss < best_loss:
        bestmodel_save_path = os.path.join(eval_dir, 'bestmodel_%d' % (save_no % 3))  # this is where checkpoints of best models are saved
        if best_loss is not None:
            logger.info(
                '[INFO] Found new best model with %.6f running_avg_loss. The original loss is %.6f, Saving to %s',
                float(running_avg_loss), float(best_loss), bestmodel_save_path)
        else:
            logger.info(
                '[INFO] Found new best model with %.6f running_avg_loss. The original loss is None, Saving to %s',
                float(running_avg_loss), bestmodel_save_path)
        with open(bestmodel_save_path, 'wb') as f:
            torch.save(model_class.state_dict(), f)
        best_loss = running_avg_loss
        non_descent_cnt = 0
        save_no += 1
    else:
        non_descent_cnt += 1

    if best_f is None or best_f < F:
        bestmodel_save_path = os.path.join(eval_dir, 'bestFmodel')  # this is where checkpoints of best models are saved
        if best_f is not None:
            logger.info('[INFO] Found new best model with %.6f F. The original F is %.6f, Saving to %s', float(F),
                        float(best_f), bestmodel_save_path)
        else:
            logger.info('[INFO] Found new best model with %.6f F. The original F is None, Saving to %s', float(F),
                        bestmodel_save_path)
        with open(bestmodel_save_path, 'wb') as f:
            torch.save(model_class.state_dict(), f)
        best_f = F

    return best_loss, best_f, non_descent_cnt, save_no


def main():
    parser = argparse.ArgumentParser(description='ExtComAbs Model')
    # Where to find data
    parser.add_argument('--data_dir', type=str, default='datasets/cnndm',help='The dataset directory.')
    parser.add_argument('--cache_dir', type=str, default='cache/cnndm',help='The processed dataset directory')
    parser.add_argument('--embedding_path', type=str, default='./Glove/glove.42B.300d.txt', help='Path expression to external word embedding.')
    parser.add_argument('--model', type=str, default='class',help='generate document or classfiler[gen_doc or class]')
    parser.add_argument('--restore_model', type=str, default='None', help='Restore model for further training. [bestmodel/bestFmodel/earlystop/None]')

    # Where to save output
    parser.add_argument('--save_root', type=str, default='save/', help='Root directory for all model.')
    parser.add_argument('--log_root', type=str, default='log/', help='Root directory for all logging.')

    # Hyperparameters
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use. [default: 0]')
    parser.add_argument('--cuda', action='store_true', default=False, help='GPU or CPU [default: False]')
    parser.add_argument('--vocab_size', type=int, default=10,help='Size of vocabulary. [default: 50000]')
    parser.add_argument('--n_epochs', type=int, default=3, help='Number of epochs [default: 20]')
    parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size [default: 32]')
    parser.add_argument('--word_embedding', action='store_true', default=True, help='whether to use Word embedding [default: True]')
    parser.add_argument('--word_emb_dim', type=int, default=4, help='Word embedding size [default: 256],Glove dim is 300')
    parser.add_argument('--embed_train', action='store_true', default=False,help='whether to train Word embedding [default: False]')
    parser.add_argument('--n_head', type=int, default=1, help='multihead attention number [default: 8]')
    parser.add_argument('--sent_max_len', type=int, default=5,help='max length of sentences (max source text sentence tokens)[default:100]')
    parser.add_argument('--doc_max_timesteps', type=int, default=3,help='max length of documents (max timesteps of documents)[default:50]')

    # Training
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_descent', action='store_true', default=True, help='learning rate descent')
    parser.add_argument('--grad_clip', action='store_true', default=True, help='for gradient clipping')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='for gradient clipping max gradient normalization')
    parser.add_argument('-m', type=int, default=3, help='decode summary length')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.set_printoptions(threshold=50000)

    # File paths
    DATA_FILE = os.path.join(args.data_dir, "train.label.jsonl2")
    VALID_FILE = os.path.join(args.data_dir, "val.label.jsonl2")
    VOCAL_FILE = os.path.join(args.cache_dir, "vocab")
    LOG_PATH = args.log_root

    # train_log setting
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_PATH, "train_" + now_time)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Pytorch %s", torch.__version__)
    logger.info("[INFO] Create Vocab, vocab path is %s", VOCAL_FILE)
    # 创建词汇表
    vocab = Vocab(VOCAL_FILE, args.vocab_size)
    embed = torch.nn.Embedding(vocab.size(), args.word_emb_dim, padding_idx=0)
    # 加载预训练的Embedding权重
    # if args.word_embedding:
    if False:
        embed_loader = Word_Embedding(args.embedding_path, vocab)
        vectors = embed_loader.load_my_vecs(args.word_emb_dim)
        pretrained_weight = embed_loader.add_unknown_words_by_avg(vectors, args.word_emb_dim)
        # copy预训练的权重参数
        embed.weight.data.copy_(torch.Tensor(pretrained_weight))
        # 是否对Embedding进行train
        embed.weight.requires_0grad = args.embed_train

    logger.info(args)

    dataset = ExampleSet(DATA_FILE, vocab, args.doc_max_timesteps, args.sent_max_len)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    del dataset

    valid_dataset= ExampleSet(VALID_FILE, vocab, args.doc_max_timesteps, args.sent_max_len)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    if args.model == "gen_doc":
        model = GengrateDocument(args, embed)
        logger.info("[MODEL] GengrateDocument ")
        if args.cuda:
            model.to(torch.device("cuda"))
            logger.info("[INFO] Use cuda")
        setup_training(model, train_loader, valid_loader, valid_dataset, args , vocab)
    else:
        model_gen = GengrateDocument(args, embed)
        model_class = Classfiler(args, embed)
        logger.info("[MODEL] Classfiler ")
        if args.cuda:
            model_gen.to(torch.device("cuda"))
            model_class.to(torch.device("cuda"))
            logger.info("[INFO] Use cuda")
        run_training_class(model_gen, model_class, train_loader, valid_loader, valid_dataset, args , vocab)

if __name__ == '__main__':
    main()
