import os
import torch
from models import models as networks
from models.models_CNN import resnet18, ResNet, BasicBlock
from modules import DTW_align, GreedyCTCDecoder, AttrDict, RMSELoss, save_checkpoint
from modules import mel2wav_vocoder, perform_STT
from utils import data_denorm, word_index
import torch.nn as nn
import torch.nn.functional as F
from NeuroTalkDataset import myDataset
import time
import torch.optim.lr_scheduler
import numpy as np
import torchaudio
from torchmetrics import CharErrorRate
from sklearn.metrics import f1_score
import json
import argparse
import wavio
from torch.utils.tensorboard import SummaryWriter
from models.ESN_py.config import parse_args
from models.mamba_config import mamba_args
from soft_dtw import SoftDTW

    
def train(args, train_loader, models, criterions, optimizers, epoch, trainValid=True, inference=False):
    '''
    :param args: general arguments
    :param train_loader: loaded for training/validation/test dataset
    :param model: model
    :param criterion: loss function
    :param optimizer: optimization algo, such as ADAM or SGD
    :param epoch: epoch number
    :return: losses
    '''
    (optimizer_g, optimizer_d) = optimizers
    
    # switch to train mode
    assert type(models) == tuple, "More than two models should be inputed (generator and discriminator)"

    epoch_loss_g = []
    epoch_loss_d = []
    
    epoch_acc_g = []
    epoch_acc_d = []
    

    total_batches = len(train_loader)
    
    for i, (input, target, target_cl,  data_info) in enumerate(train_loader):    

        print("\rBatch [%5d / %5d]"%(i,total_batches), sep=' ', end='', flush=True)
        
        input = input.cuda()
        target = target.cuda()
        target_cl = target_cl.cuda()
        # voice = torch.squeeze(voice,dim=-1).cuda()
        labels = torch.argmax(target_cl,dim=1)

        
        # extract unseen
        idx_seen=[]
        for j in range(len(labels)):
            idx_seen.append(j)
        
        
        input = input[idx_seen]
        #print(f"input's shape : {input.shape}")
        target = target[idx_seen]
        #print(f"target's shape : {target.shape}")
        target_cl = target_cl[idx_seen]
        # voice = voice[idx_seen]
        labels = labels[idx_seen]
        data_info = [data_info[0][idx_seen],data_info[1][idx_seen]]
        
        # # need to remove
        # models = (model_g, model_d, vocoder, model_STT, decoder_STT)
        # criterions = (criterion_recon, criterion_ctc, criterion_adv, criterion_cl, CER)
        # trainValid = True
        
        # general training         
        if len(input) != 0:
            # train generator
            mel_out, e_loss_g, e_acc_g = train_G(args, 
                                                 input, target,  labels,  
                                                 models, criterions, optimizer_g, 
                                                 data_info, 
                                                 trainValid)
            epoch_loss_g.append(e_loss_g)
            epoch_acc_g.append(e_acc_g)
        
            # train discriminator
            e_loss_d, e_acc_d = train_D(args, 
                                        mel_out, target, target_cl, labels,
                                        models, criterions, optimizer_d, 
                                        trainValid)
            epoch_loss_d.append(e_loss_d)
            epoch_acc_d.append(e_acc_d)
    

    epoch_loss_g = np.array(epoch_loss_g).reshape(1,-1)
    epoch_acc_g = np.array(epoch_acc_g).reshape(1,-1)
    epoch_loss_d = np.array(epoch_loss_d).reshape(1,-1)
    epoch_acc_d = np.array(epoch_acc_d).reshape(1,-1)
    
    args.loss_g = sum(epoch_loss_g[:,0]) / len(epoch_loss_g[:,0])
    args.loss_g_recon = sum(epoch_loss_g[:,1]) / len(epoch_loss_g[:,1])
    args.loss_g_valid = sum(epoch_loss_g[:,2]) / len(epoch_loss_g[:,2])
    args.loss_g_dtw = sum(epoch_loss_g[:,3]) / len(epoch_loss_g[:,3])
    args.acc_g_valid = sum(epoch_acc_g[:,0]) / len(epoch_acc_g[:,0])
    args.f1_recon = sum(epoch_acc_g[:,1]) / len(epoch_acc_g[:,1])

    
    args.loss_d = sum(epoch_loss_d[:,0]) / len(epoch_loss_d[:,0])
    args.loss_d_valid = sum(epoch_loss_d[:,1]) / len(epoch_loss_d[:,1])
    args.loss_d_cl = sum(epoch_loss_d[:,2]) / len(epoch_loss_d[:,2])
    args.acc_d_real = sum(epoch_acc_d[:,0]) / len(epoch_acc_d[:,0])
    args.acc_d_fake = sum(epoch_acc_d[:,1]) / len(epoch_acc_d[:,1])
    args.acc_cl_real = sum(epoch_acc_d[:,2]) / len(epoch_acc_d[:,2])
    args.acc_cl_fake = sum(epoch_acc_d[:,3]) / len(epoch_acc_d[:,3])
    

    
    # tensorboard
    if trainValid:
        tag = 'train'
    else:
        tag = 'valid'
        
    if not inference:
        args.writer.add_scalar("Loss_G/{}".format(tag), args.loss_g, epoch)
        args.writer.add_scalar("f1/{}".format(tag), args.f1_recon, epoch)

        args.writer.add_scalar("Loss_D_valid/{}".format(tag), args.loss_d_valid, epoch)
        args.writer.add_scalar("Loss_D_cl/{}".format(tag), args.loss_d_cl, epoch)
        
        args.writer.add_scalar("Loss_G_recon/{}".format(tag), args.loss_g_recon, epoch)
        args.writer.add_scalar("Loss_G_valid/{}".format(tag), args.loss_g_valid, epoch)
        args.writer.add_scalar("Loss_G_dtw/{}".format(tag), args.loss_g_dtw, epoch)
        
        args.writer.add_scalar("ACC_D_real/{}".format(tag), args.acc_d_real, epoch)
        args.writer.add_scalar("ACC_D_fake/{}".format(tag), args.acc_d_fake, epoch)


    print('\n[%3d/%3d] f1-recon: %.4f / ACC_R: %.4f ACC_F: %.4f / g-RMSE: %.4f g-lossValid: %.4f g-lossCLS: %.4f' 
          % (i, total_batches, 
             args.f1_recon, 
             args.acc_d_real, args.acc_d_fake, 
             args.loss_g_recon, args.loss_g_valid, args.loss_g_dtw))
        
        
    return (args.loss_g, args.loss_g_recon, args.loss_g_valid, args.loss_g_dtw, args.acc_g_valid, args.f1_recon, 
            args.loss_d, args.acc_d_real, args.acc_d_fake)


def train_G(args, input, target,  labels, models, criterions, optimizer_g, data_info, trainValid):    
    (model_g, model_d, classifier) = models
    (criterion_recon, criterion_dtw, criterion_adv, _) =  criterions
    
    if trainValid:
        model_g.train()
        model_d.train()
        classifier.eval()
    else:
        model_g.eval()
        model_d.eval()
        classifier.eval()
    
    # Adversarial ground truths 1:real, 0: fake
    valid = torch.ones((len(input), 1), dtype=torch.float32).cuda()
    
    ###############################
    # Train Generator
    ###############################
    
    if trainValid:
        for p in model_g.parameters():
            p.requires_grad_(True)   # unfreeze G
        for p in model_d.parameters():
            p.requires_grad_(False)  # freeze D
        for p in classifier.parameters():
            p.requires_grad_(False)  # freeze vocoder

        # set zero grad    
        optimizer_g.zero_grad()
        
        # Run Generator
        output = model_g(input.permute(0,2,1))
    else:
        with torch.no_grad():
            # run generator
            output = model_g(input.permute(0,2,1))

    #print(f"output's shape: {output.shape}")
    # DTW
    mel_out = output.clone()
    #mel_out = DTW_align(mel_out, target)
    
    # Run Discriminator
    g_valid, _ = model_d(mel_out)
    
    # generator loss
    loss_recon = criterion_recon(mel_out, target)
    #loss_recon = loss_recon.mean()
    # GAN loss
    loss_valid = criterion_adv(g_valid, valid)
    
    # accuracy    args.l_g = h_g.l_g
    acc_g_valid = (g_valid.round() == valid).float().mean()
    

    #target_denorm = data_denorm(target, data_info[0], data_info[1])
    #output_denorm = data_denorm(mel_out, data_info[0], data_info[1])
    target_denorm = target
    output_denorm = mel_out
    
    #reshape
    output_reshape = output_denorm.reshape(-1,1,80,172)
    
    # target
    pred_labels = classifier(output_reshape)
    
    # DTW Loss
    loss_dtw = criterion_dtw(mel_out, target)
    loss_dtw = loss_dtw.mean()
    # ClS loss
    #loss_cls = criterion_cls(pred_labels, labels)
    args.l_g[2] = 1e-4
    # total generator loss
    loss_g = args.l_g[0] * loss_recon - args.l_g[1] * loss_valid + args.l_g[2] * loss_cls
    
    # Calculate the L2 norm of the model parameters
    #l2_norm = sum(p.pow(2.0).sum() for p in model_g.parameters())

    # Add the L2 regularization term to the loss
    #loss_g += args.lambda_reg * l2_norm

    _, predicted = torch.max(pred_labels, 1)
    f1_recon = f1_score(predicted.detach().cpu(), labels.detach().cpu(), average='macro')

    if trainValid:
        loss_g.backward() 
        optimizer_g.step()
    
    e_loss_g = (loss_g.item(), loss_recon.item(), loss_valid.item(), loss_cls.item())
    e_acc_g = (acc_g_valid.item(), f1_recon.item())
    
    return mel_out, e_loss_g, e_acc_g
      
    
def train_D(args, mel_out, target, target_cl, labels, models, criterions, optimizer_d, trainValid):
    
    (_, model_d, _) = models
    (_, _, criterion_adv, criterion_cl) =  criterions

    if trainValid:
        model_d.train()
    else:
        model_d.eval()
    
    # Adversarial ground truths 1:real, 0: fake
    valid = torch.ones((len(mel_out), 1), dtype=torch.float32).cuda()
    fake = torch.zeros((len(mel_out), 1), dtype=torch.float32).cuda()
    
    ###############################
    # Train Discriminator
    ###############################
    
    if trainValid:
        if args.pretrain and args.prefreeze:
            for total_ct, _ in enumerate(model_d.children()):
                ct=0
            for ct, child in enumerate(model_d.children()):
                if ct > total_ct-1: # unfreeze classifier 
                    for param in child.parameters():
                        param.requires_grad = True  # unfreeze D    
        else:
            for p in model_d.parameters():
                p.requires_grad_(True)  # unfreeze D   
                
        # set zero grad
        optimizer_d.zero_grad()

    # run model cl
    real_valid, real_cl = model_d(target)
    fake_valid, fake_cl = model_d(mel_out.detach())

    loss_d_real_valid = criterion_adv(real_valid, valid)
    loss_d_fake_valid = criterion_adv(fake_valid, fake)
    loss_d_real_cl = criterion_cl(real_cl, target_cl)
    
    loss_d_valid = 0.5 * (loss_d_real_valid + loss_d_fake_valid)
    loss_d_cl = loss_d_real_cl
    
    loss_d = args.l_d[0] * loss_d_cl + args.l_d[1] * loss_d_valid
    
    # accuracy
    acc_d_real = (real_valid.round() == valid).float().mean()
    acc_d_fake = (fake_valid.round() == fake).float().mean()
    preds_real = torch.argmax(real_cl,dim=1)
    acc_cl_real = (preds_real == labels).float().mean()
    preds_fake = torch.argmax(fake_cl,dim=1)
    acc_cl_fake = (preds_fake == labels).float().mean()
    
    if trainValid:
        loss_d.backward()
        optimizer_d.step()

    e_loss_d = (loss_d.item(), loss_d_valid.item(), loss_d_cl.item())
    e_acc_d = (acc_d_real.item(), acc_d_fake.item(), acc_cl_real.item(), acc_cl_fake.item())
    
    return e_loss_d, e_acc_d


def saveData(args, test_loader, models, epoch, losses):
    
    model_g = models[0].eval()
    # model_d = models[1].eval()
    vocoder = models[2].eval()
    model_STT = models[3].eval()
    decoder_STT = models[4]

    input, target, target_cl,  data_info = next(iter(test_loader))   
    
    input = input.cuda()
    target = target.cuda()
    voice = torch.squeeze(voice,dim=-1).cuda()
    labels = torch.argmax(target_cl,dim=1)    
    
    with torch.no_grad():
        # run the mdoel
        output = model_g(input)
    
    mel_out = DTW_align(output, target)
    output_denorm = data_denorm(mel_out, data_info[0], data_info[1])
    
    wav_recon = mel2wav_vocoder(torch.unsqueeze(output_denorm[0],dim=0), vocoder, 1)
    wav_recon = torch.reshape(wav_recon, (len(wav_recon),wav_recon.shape[-1]))
    
    wav_recon = torchaudio.functional.resample(wav_recon, args.sample_rate_mel, args.sample_rate_STT)  
    if wav_recon.shape[1] !=  voice.shape[1]:
        p = voice.shape[1] - wav_recon.shape[1]
        p_s = p//2
        p_e = p-p_s
        wav_recon = F.pad(wav_recon, (p_s,p_e))
        
    ##### STT Wav2Vec 2.0
    gt_label = args.word_label[labels[0].item()]
    
    transcript_recon = perform_STT(wav_recon, model_STT, decoder_STT, gt_label, 1)
    
    # save
    wav_recon = np.squeeze(wav_recon.cpu().detach().numpy())
    
    str_tar = args.word_label[labels[0].item()].replace("|", ",")
    str_tar = str_tar.replace(" ", ",")
    
    str_pred = transcript_recon[0].replace("|", ",")
    str_pred = str_pred.replace(" ", ",")
    
    title = "Tar_{}-Pred_{}".format(str_tar, str_pred)
    wavio.write(args.savevoice + '/e{}_{}.wav'.format(str(str(epoch)), title), wav_recon, args.sample_rate_STT, sampwidth=2)


def main(args):
    device = torch.device(f'cuda:{args.gpuNum[0]}' if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device: {} '.format(torch.cuda.current_device())) # check
    print('The number of available GPU:{}'.format(torch.cuda.device_count()))
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # define generator
    config_file = os.path.join(args.model_config, 'config_g.json')
    with open(config_file) as f:
        data = f.read()
    #json_config = json.loads(data)
    #h_g = AttrDict(json_config)
    h_g = mamba_args()
    #h_g = parse_args()
    model_g = networks.Generator_Mamba(h_g).cuda()
    
    args.sample_rate_mel = args.sampling_rate
    
    # define discriminator
    config_file = os.path.join(args.model_config, 'config_d.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h_d = AttrDict(json_config)
    model_d = networks.Discriminator(h_d).cuda()
    
    # classifier
    classifier = ResNet(BasicBlock, [2,2,2,2], 13)
    classifier = torch.load(args.decoder_pre+args.sub+'/'+"pre_trained/"+'CNN_'+args.sub+'.pt').cuda()

    
    #model_g_wout = [model_g.Wout]
    
    # Parallel setting
    model_g = nn.DataParallel(model_g, device_ids=args.gpuNum)
    model_d = nn.DataParallel(model_d, device_ids=args.gpuNum)
    classifier = nn.DataParallel(classifier, device_ids=args.gpuNum)


    # loss function
    criterion_recon = RMSELoss().cuda() #SoftDTW(gamma=0.3, normalize=True) #RMSELoss().cuda()
    criterion_adv = nn.BCELoss().cuda()
    criterion_dtw = SoftDTW(gamma=0.3, normalize=True)#nn.CrossEntropyLoss().cuda()
    criterion_cl = nn.CrossEntropyLoss().cuda()

    args.lr_g = 1e-4

    # optimizer
    optimizer_g = torch.optim.AdamW(model_g.parameters(), lr=args.lr_g, betas=(0.8, 0.99), weight_decay=0.01)
    optimizer_d = torch.optim.AdamW(model_d.parameters(), lr=args.lr_d, betas=(0.8, 0.99), weight_decay=0.01)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=args.lr_g_decay, last_epoch=-1)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=args.lr_d_decay, last_epoch=-1)

   # create the directory if not exist
    if not os.path.exists(args.logDir):
        os.mkdir(args.logDir)
        
    subDir = os.path.join(args.logDir, args.sub)
    if not os.path.exists(subDir):
        os.mkdir(subDir)        
        
    saveDir = os.path.join(args.logDir, args.sub, args.task)
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

    logsDir = os.path.join(args.logDir, args.sub, args.task, "logs")
    if not os.path.exists(logsDir):
        os.mkdir(logsDir)
    
    savemodelDir = os.path.join(args.logDir, args.sub, args.task, "savemodel")
    if not os.path.exists(savemodelDir):
        os.mkdir(savemodelDir)
        
    # args.savevoice = saveDir + '/epovoice'
    # if not os.path.exists(args.savevoice):
    #     os.mkdir(args.savevoice)
    args.logs = saveDir + '/logs/' + str(args.batch_size) + '/'
    if not os.path.exists(args.logs):
        os.mkdir(args.logs)
    
    n_num = len(os.listdir(args.logs)) 
    save_model_path = f"{n_num+1}_lr_g{args.lr_g}_l_g0{args.l_g[0]}_lg1{args.l_g[1]}_lg2{args.l_g[2]}_lambda{args.lambda_reg}/"

    args.savemodel = saveDir + f'/savemodel/{save_model_path}'
    if not os.path.exists(args.savemodel):
        os.mkdir(args.savemodel)
        

        
    # Load trained model
    start_epoch = 0
    if args.pretrain:
        loc_g = os.path.join(args.trained_model, args.sub, 'BEST_checkpoint_g.pt')
        loc_d = os.path.join(args.trained_model, args.sub, 'BEST_checkpoint_d.pt')

        if os.path.isfile(loc_g):
            print("=> loading checkpoint '{}'".format(loc_g))
            checkpoint_g = torch.load(loc_g, map_location='cpu')
            model_g.load_state_dict(checkpoint_g['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(loc_g))

        if os.path.isfile(loc_d):
            print("=> loading checkpoint '{}'".format(loc_d))
            checkpoint_d = torch.load(loc_d, map_location='cpu')
            model_d.load_state_dict(checkpoint_d['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(loc_d))

    if args.resume:
        loc_g = os.path.join(args.savemodel, 'checkpoint_g.pt')
        loc_d = os.path.join(args.savemodel, 'checkpoint_d.pt')

        if os.path.isfile(loc_g):
            print("=> loading checkpoint '{}'".format(loc_g))
            checkpoint_g = torch.load(loc_g, map_location='cpu')
            model_g.load_state_dict(checkpoint_g['state_dict'])
            start_epoch = checkpoint_g['epoch'] + 1
        else:
            print("=> no checkpoint found at '{}'".format(loc_g))

        if os.path.isfile(loc_d):
            print("=> loading checkpoint '{}'".format(loc_d))
            checkpoint_d = torch.load(loc_d, map_location='cpu')
            model_d.load_state_dict(checkpoint_d['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(loc_d))

    # Tensorboard setting
    
    log_name = f"{n_num+1}_lr_g{args.lr_g}_l_g0{args.l_g[0]}_lg1{args.l_g[1]}_lg2{args.l_g[2]}_lambda{args.lambda_reg}"
    args.writer = SummaryWriter(args.logs+log_name)
    
    # Data loader define
    generator = torch.Generator().manual_seed(args.seed)
    if args.task.find("Spoken") == 0:
        args.ta = "sp"
    elif args.task.find("Imagined") == 0:
        args.ta = "im"

    trainset = myDataset(mode=0, data=args.dataLoc+'/'+args.sub, task=args.task, recon=args.ta+"_"+args.recon)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=4*len(args.gpuNum), pin_memory=True)
    
    valset = myDataset(mode=2, data=args.dataLoc+'/'+args.sub, task=args.task, recon=args.ta+"_"+args.recon)
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=4*len(args.gpuNum), pin_memory=True)

    epoch = start_epoch
    lr_g = 0
    lr_d = 0
    best_loss = 1000
    is_best = False
    epochs_since_improvement = 0
    
    for epoch in range(start_epoch, args.max_epochs):
        
        start_time = time.time()
        
        for param_group in optimizer_g.param_groups:
            lr_g = param_group['lr']
        for param_group in optimizer_d.param_groups:
            lr_d = param_group['lr']

        scheduler_g.step(epoch)
        scheduler_d.step(epoch)

        print("Epoch : %d/%d" %(epoch, args.max_epochs) )
        print("Learning rate for G: %.9f" %lr_g)
        print("Learning rate for D: %.9f" %lr_d)

        Tr_losses = train(args, train_loader, 
                          (model_g, model_d, classifier), 
                          (criterion_recon, criterion_dtw, criterion_adv, criterion_cl), 
                          (optimizer_g, optimizer_d), 
                          epoch,
                          True) 

        
        Val_losses = train(args, val_loader, 
                           (model_g, model_d, classifier), 
                           (criterion_recon, criterion_dtw, criterion_adv, criterion_cl), 
                           ([],[]), 
                           epoch,
                           False)
        
        # Save checkpoint
        state_g = {'arch': str(model_g),
                 'state_dict': model_g.state_dict(),
                 'epoch': epoch,
                 'optimizer_state_dict': optimizer_g.state_dict()}
        
        state_d = {'arch': str(model_d),
                 'state_dict': model_d.state_dict(),
                 'epoch': epoch,
                 'optimizer_state_dict': optimizer_d.state_dict()}
        
        # Did validation loss improve?
        loss_total =  Val_losses[3] # default is 0
        is_best = loss_total < best_loss
        best_loss = min(loss_total, best_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        save_checkpoint(state_g, is_best, args.savemodel, 'checkpoint_g.pt')
        save_checkpoint(state_d, is_best, args.savemodel, 'checkpoint_d.pt')

        # saveData(args, val_loader, (model_g, model_d, vocoder, model_STT, decoder_STT), epoch, (Tr_losses,Val_losses))

        time_taken = time.time() - start_time
        print("Time: %.2f\n"%time_taken)
        
    args.writer.flush()

if __name__ == '__main__':

    dataDir = './dataset'
    logDir = './TrainResult'
    
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--vocoder_pre', type=str, default='./pretrained_model/UNIVERSAL_V1/g_02500000', help='pretrained vocoder file path')
    parser.add_argument('--trained_model', type=str, default='./pretrained_model', help='trained model for G & D folder path')
    parser.add_argument('--model_config', type=str, default='./models', help='config for G & D folder path')
    parser.add_argument('--dataLoc', type=str, default=dataDir)
    parser.add_argument('--config', type=str, default='./config.json')
    parser.add_argument('--logDir', type=str, default=logDir)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--prefreeze', type=bool, default=False)
    parser.add_argument('--gpuNum', type=list, default=[0])
    parser.add_argument('--batch_size', type=int, default=26)
    parser.add_argument('--sub', type=str, default='sub2')
    parser.add_argument('--task', type=str, default='SpokenEEG')
    parser.add_argument('--recon', type=str, default='Y_mel')
    parser.add_argument('--unseen', type=str, default='stop')
    parser.add_argument('--decoder_pre', type=str, default='./Patients/preprocessed/')
    parser.add_argument('--subject_name', type=str, default='CYJ')
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    #args.lambda_reg = 0.1
    main(args)        
    
    
    
