import json
import torch
import utils.utils as utils
from vsa_clevr.vsa_reasoner import VSAReasoner
from vsa_clevr.vsa_scene_parser import VSASceneParser 

DESCR = {
    'attribute': ['attribute', 'color', 'size', 'shape', 'material', 'coordinates'],
    'color': ['purple', 'blue', 'brown', 'cyan', 'yellow', 'red', 'gray', 'green'],
    'size': ['small', 'large'],
    'shape': ['sphere', 'cylinder', 'cube'],
    'material': ['metal', 'rubber'],
    'coordinates': ['x_coord', 'y_coord', 'z_coord']
}

HD_DIM = 30000
VSA_TYPE = 'polar'
THR = 6

class Trainer():
    """Trainer"""

    def __init__(self, opt, train_loader, val_loader, model, executor):
        self.opt = opt
        self.vsa = opt.vsa
        self.vsa_parser_train = VSASceneParser('../data/raw/CLEVR_v1.0/scenes/CLEVR_train_scenes.json', dim=HD_DIM, vsa_type=VSA_TYPE, thr=THR, descr=DESCR)
        self.vsa_parser_val = VSASceneParser('../data/raw/CLEVR_v1.0/scenes/CLEVR_val_scenes.json', dim=HD_DIM, vsa_type=VSA_TYPE, thr=THR, descr=DESCR)
        self.reinforce = opt.reinforce
        self.ppo = opt.ppo
        self.reward_decay = opt.reward_decay
        self.entropy_factor = opt.entropy_factor
        self.num_iters = opt.num_iters
        self.run_dir = opt.run_dir
        self.display_every = opt.display_every
        self.checkpoint_every = opt.checkpoint_every
        self.visualize_training = opt.visualize_training
        if opt.dataset == 'clevr':
            self.vocab = utils.load_vocab(opt.clevr_vocab_path)
        elif opt.dataset == 'clevr-humans':
            self.vocab = utils.load_vocab(opt.human_vocab_path)
        else:
            raise ValueError('Invalid dataset')

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.executor = executor
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.seq2seq.parameters()),
                                          lr=opt.learning_rate)

        self.stats = {
            'train_losses': [],
            'train_advantages': [],
            'train_batch_accs': [],
            'train_accs_ts': [],
            'val_losses': [],
            'val_accs': [],
            'val_accs_ts': [],
            'best_val_acc': -1,
            'model_t': 0
        }
        if opt.visualize_training:
            from utils.logger import Logger
            self.logger = Logger('%s/logs' % opt.run_dir)

    def train(self):
        training_mode = 'reinforce' if self.reinforce else 'seq2seq'
        if self.ppo:
            training_mode = 'PPO'
        print('| start training %s, running in directory %s' % (training_mode, self.run_dir))
        t = 0
        epoch = 0
        baseline = 0
        while t < self.num_iters:
            epoch += 1
            for x, y, ans, idx in self.train_loader:
                t += 1
                loss, reward = None, None
                self.model.set_input(x, y)
                if not self.ppo:
                    self.optimizer.zero_grad()
                if self.reinforce:
                    pred = self.model.reinforce_forward()
                    reward = self.get_batch_reward(pred, ans, idx, 'train', self.vsa)
                    baseline = reward * (1 - self.reward_decay) + baseline * self.reward_decay
                    advantage = reward - baseline
                    self.model.set_reward(advantage)
                    self.model.reinforce_backward(self.entropy_factor)
                elif self.ppo:
                    pred = self.model.ppo_forward()
                    reward = self.get_batch_reward(pred, ans, idx, 'train', self.vsa)
                    baseline = reward * (1 - self.reward_decay) + baseline * self.reward_decay
                    advantage = reward - baseline
                    self.model.set_reward_ppo(advantage)
                    if t % 1000 == 0:
                        self.model.ppo_backward(self.optimizer, self.entropy_factor)
                else:
                    loss = self.model.supervised_forward()
                    self.model.supervised_backward()
                if not self.ppo:
                    self.optimizer.step()

                if t % self.display_every == 0:
                    if self.reinforce or self.ppo:
                        self.stats['train_batch_accs'].append(reward)
                        self.stats['train_advantages'].append(advantage)
                        self.log_stats('training batch reward', reward, t)
                        self.log_stats('training batch advantage', advantage, t)
                        print('| iteration %d / %d, epoch %d, reward %f' % (t, self.num_iters, epoch, reward))
                    else:
                        self.stats['train_losses'].append(loss)
                        self.log_stats('training batch loss', loss, t)
                        print('| iteration %d / %d, epoch %d, loss %f' % (t, self.num_iters, epoch, loss))
                    self.stats['train_accs_ts'].append(t)

                if t % self.checkpoint_every == 0 or t >= self.num_iters:
                    print('| checking validation accuracy')
                    val_acc = self.check_val_accuracy()
                    print('| validation accuracy %f' % val_acc)
                    if val_acc >= self.stats['best_val_acc']:
                        print('| best model')
                        self.stats['best_val_acc'] = val_acc
                        self.stats['model_t'] = t
                        self.model.save_checkpoint('%s/checkpoint_best.pt' % self.run_dir)
                        self.model.save_checkpoint('%s/checkpoint_iter%08d.pt' % (self.run_dir, t))
                    if not self.reinforce and not self.ppo:
                        val_loss = self.check_val_loss()
                        print('| validation loss %f' % val_loss)
                        self.stats['val_losses'].append(val_loss)
                        self.log_stats('val loss', val_loss, t)
                    self.stats['val_accs'].append(val_acc)
                    self.log_stats('val accuracy', val_acc, t)
                    self.stats['val_accs_ts'].append(t)
                    self.model.save_checkpoint('%s/checkpoint.pt' % self.run_dir)
                    with open('%s/stats_tfevents.json' % self.run_dir, 'w') as fout:
                        json.dump(self.stats, fout)
                    self.log_params(t)

                if t >= self.num_iters:
                    break

    def check_val_loss(self):
        loss = 0
        t = 0
        for x, y, _, _ in self.val_loader:
            self.model.set_input(x, y)
            loss += self.model.supervised_forward()
            t += 1
        return loss / t if t is not 0 else 0

    def check_val_accuracy(self):
        reward = 0
        t = 0
        for x, y, ans, idx in self.val_loader:
            self.model.set_input(x, y)
            pred = self.model.parse()
            reward += self.get_batch_reward(pred, ans, idx, 'val')
            t += 1
        reward = reward / t if t is not 0 else 0
        return reward 

    def get_batch_reward(self, programs, answers, image_idxs, split, vsa=False):
        pg_np = programs.numpy()
        ans_np = answers.numpy()
        idx_np = image_idxs.numpy()
        reward = 0

        if vsa:
            for i in range(pg_np.shape[0]):
                if split == 'train':
                    scene = self.vsa_parser_train.parse(idx_np[i])
                else:
                    scene = self.vsa_parser_val.parse(idx_np[i])

                executor = VSAReasoner(scene)

                pred, _ = executor.run(pg_np[i], scene)
                ans = executor.vocab['answer_idx_to_token'][ans_np[i]]

                if pred == ans:
                    reward += 1.0
        else:
            for i in range(pg_np.shape[0]):
                pred, _ = self.executor.run(pg_np[i], idx_np[i], split)
                ans = self.vocab['answer_idx_to_token'][ans_np[i]]
                if pred == ans:
                    reward += 1.0

        reward /= pg_np.shape[0]
        return reward

    def log_stats(self, tag, value, t):
        if self.visualize_training and self.logger is not None:
            self.logger.scalar_summary(tag, value, t)

    def log_params(self, t):
        if self.visualize_training and self.logger is not None:
            for tag, value in self.model.seq2seq.named_parameters():
                tag = tag.replace('.', '/')
                self.logger.histo_summary(tag, self._to_numpy(value), t)
                if value.grad is not None:
                    self.logger.histo_summary('%s/grad' % tag, self._to_numpy(value.grad), t)

    def _to_numpy(self, x):
        return x.data.cpu().numpy()