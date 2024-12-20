import torch 
import torch.nn as nn 

import torchmetrics 
import tqdm 
import warnings 


from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel , BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer  , BpeTrainer

from pathlib import Path 

from torch.utils.data import Dataset , DataLoader , random_split
from dataset import BilingualDataset , causal_mask
from model import build_transformer 
from config import get_config, get_weights_file_path, latest_weights_file_path

from torch.utils.tensorboard import SummaryWriter 

# dataset = [
#     {"translation": {"en": "Hello world!", "fr": "Bonjour le monde!"}},
#     {"translation": {"en": "Deep learning is amazing.", "fr": "L'apprentissage profond est incroyable."}},
#     {"translation": {"en": "Transformers are powerful.", "fr": "Les Transformers sont puissants."}},
# ]

def get_all_sentences(ds , lang ):
    # typically the data looks like above so if we give the language it will retrive tha  sentnces of the given lang 
    # genrally for handling the large data we will use the yeild statemnt 
    # return or yeild is both same but return return the entire data at once but yeild will get the data step by step  
    
    for item in ds :
        yield item['translation'][lang]

#  *** potntial change is you can add the post tokenizeation templating ie add sos eos ***        
def get_or_build_tokenizer(config , ds , lang ):
    """
    Get or build a tokenizer based on the given dataset and language.
    If a saved tokenizer exists, load it; otherwise, build and save a new one.
    
    Args:
        config (dict): Configuration dictionary containing the tokenizer file path.
        ds (Dataset): Dataset containing the sentences to train on.
        lang (str): Language identifier (used for tokenizer file naming).
    
    Returns:
        Tokenizer: Trained or loaded tokenizer.
    """
    # we will check the tokenizer path  is there or not 
    # if path not there we will build the tokenizer 
    # for bilding the tokenizer we will use the tokenizers lib of transformers 
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # step1 define the tokenizermodel 
        
        tokenizer = Tokenizer(WordLevel(unk_token= "[UNK]"))
        # this is for the byte pair version 
        # tokenizer = Tokenizer(BPE(unk_token = "[UNK]"))
        
        # step2 set the pretrainer
        # the pre trainer will split the sentenses given intot the 
        # based on given type/condition like white space or bitpair
        #  or unigram variant 
        
        tokenizer.pre_tokenizer = Whitespace()
        
        # step3 CREATE A TRAINER this trainer will have the 
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"] , min_frequency = 2 )
        # this is for the byte paiar vesion
        # trainer = BpeTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"] , min_frequency = 2 )
        
        
        # step4 trainer the model with the trainer 
        
        tokenizer.train_from_iterator( get_all_sentences(ds , lang ) , trainer = trainer  )
        # note here the sentences will come and the pretokenizers splits based on white space and then the tokens 
        # which are of min freq 2 is given the vocab status 
        # and model will learn about the tokens and assign the corresposnding id 
        
        # this one add if needed 
        # Step 5: Add post-tokenization templating for SOS and EOS
        # tokenizer.post_processor = TemplateProcessing(
        #     single="[SOS] $A [EOS]",
        #     pair="[SOS] $A [EOS] $B:1 [EOS]",
        #     special_tokens=[
        #         ("[SOS]", tokenizer.token_to_id("[SOS]")),
        #         ("[EOS]", tokenizer.token_to_id("[EOS]")),
        #     ],
        # )
        
        #  step5 save the tokenizer 
        # we will store in the path 
        tokenizer.save(str(tokenizer_path))
        
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
    return tokenizer


# ;ets make a fucntion via which we make the model 
def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model
        
# now lets make the get_dataset class the main aim of the class is :


def get_ds(config):
    
    # stpe1 : load the raw dataset 
    
    # using the load_dataset of the hugging face we can load the data
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    
    # step2 : build the tokenizer 
    tokenizer_src = get_or_build_tokenizer(config , ds_raw , config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config , ds_raw , config['lang_tgt'])
    
    # step3 make the dataset # 
    
    # now lets split the dataset for the train and the validation 
    # and then feed the raw dataset for the bilingual so that it converts 
    # according to the satandart pytorch like thing so that next we can ffed 
    # it to the dataloaders 
    
    
    train_ds_size = int( 0.9 * len(ds_raw ))
    val_ds_size = int(0.1 * len(ds_raw) )
    train_ds_raw , val_ds_raw = random_split(ds_raw  , [train_ds_size , val_ds_size ])
    
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    # step5 find the max seqlen 
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    
    # step6 make the dataloaders 
    
    train_dataloader = DataLoader(train_ds , batch_size = config['batch_size']  , shuffle = True )
    val_dataloader = DataLoader(val_ds , batch_size = 1 , shuffle = True )
    
    return train_dataloader , val_dataloader , tokenizer_src , tokenizer_tgt 


    
    
def train_model(config):
    
    # step1 we have to set the device ie if cuda or cpu 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("WE ARE USING THE DEVICE " , device )
    if(device == 'cuda' ):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    
    device = torch.device(device)
        
    # step2 make sure the weights folder exists else create one 
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    # step3 load the  dataloaders ie getting the data for traing and validation .
    # here we get it via the get_ds function 
    
    train_dataloader , val_dataloader , tokenizer_src , tokenizer_tgt = get_ds(config)
    
    # step4 : get the model use the get_model function 
    
    model = get_model(config , tokenizer_src.get_vocab_size()  , tokenizer_tgt.get_vocab_size() ).to(device)
    
    # note  maintain the consistence of adding the .to_device(device)
    
    writer = SummaryWriter(config['experiment_name'])
    
    # step5 initialize the optimizer 
    
    optimizer = torch.optim.Adam(model.parameters()  , lr = config['lr']  , eps = 1e-9 )
    
    # see by now we have initizalized themodel here we might have 3 cases 
    # 1) model yet not created and training is alsonot initizalized  
    # 2 we alreay have the model training started  and state variabels are  stored for the prev epoch 
    # 3) we trained the model and now we want to bring that weights and bias 
    
    # step6 bring the pretrianed model is alreay avialble 
    
    initial_epoch = 0 
    global_step = 0 
    
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config , preload ) if preload else None 
    
    if model_filename :
        print(f"preloading model {model_filename}" )
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] +1 
        optimizer.load_state_dict(state['optimizer_state_dict'] )
        global_step = state['global_step']
        
    else:
        print("no model to preload , starting from sratch ")
        
    # step7 we are deifing the cross entropy loss and note 
    # while caluclating the loss we should not consider the pad sequence 
    # so set the ignore_index = id of pad and also set the label smoothing ie we
    # will tel the model if label is 1 then take 0.9 to 1  and iflabel is 0 then take some value between 0 to 0.1 
     
        
    loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer_src.token_to_id('[PAD]') , label_smoothing = 0.1 ).to(device)
    
    #  start the traing loop 
    
    for epoch in range(initial_epoch , config['num_epochs'] ):
        # remove the cache in the cuda 
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        # set it in the eval mode 
        model.train()
        # set a tqdm so that we can track the process 
        batch_iterator  = tqdm(train_dataloader , desc = f" processing the epoch {epoch} ")
        
        for batch in batch_iterator:
            
            # steps that happen is 
            # forwardapss -> loss calulation -> backwaprdpass -> update parms 
            
            #  see how the batch is in the bilinguals=dataset class in the dataset.py file 
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            decoder_input = batch["decoder_input"].to(device) # (B, seq_len)
            encoder_mask =  batch["encoder_mask"].to(device) # (B, 1, 1, seq_len)
            decoder_mask =  batch["decoder_mask"].to(device) # (B, 1, seq_len, seq_len)
            
            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input , encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)
            
            
            # load the labels 
            label = batch["label"].to(device)  # (B, seq_len)
            
            
            
            
            # compare the loss using a simple cross entropy
             
            
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss" : f"{loss.item():6.3f}" })
            
            
            # log the loss 
            
            writer.add_scalar('train loss ' , loss.item() , global_step )
            writer.flush()
            
            # back propagation 
            
            loss.backward()

            # update the weights 
            
            optimizer.step()
            optimizer.zero_grad(set_to_none = True )
            
            global_step += 1
            
            
        #  now the batch are completed 
        
        # so now lets validate the info 
        # note the traing and val loop is almost similar except the gradinent update 
        
        run_validation(model , val_dataloader , tokenizer_src , tokenizer_tgt , config['seq_len']  , device ,  lambda msg: batch_iterator.write(msg), global_step, writer)
        
        # save the mdoel at the stae at this epoch 
        model_filename = get_weights_file_path(config , f"{epoch:02d}")
        torch.save({
                
                'epoch' : epoch ,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'global_step' : global_step 
            } , model_filename )


import os 

# remeber the model will behave auto regressive at the inference and non auto regressive at the training 


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    # hey first precompute the encoder input and then use it every time 
    encoder_output = model.encode(source , source_mask )
    
    #  we know the decoder input will be starting from the sos 
    # and then first word predicted and these 2 are sended and like that the loops goes on sorted
    # we are going to write this logic 
    
    decoder_input = torch.empty(1 , 1).fill_(sos_idx).type_as(source).to(device)
    
    while True:
        # we are running the loop until the max sequence is produced or the eos is found 
        if decoder_input.size(1) == max_len:
            break
        
        # build mask for target 
        
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        
        # lets calculate the output 
        
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        
        # get next token 
        
        # now lets get the new token thats found
        
        # For example, if out has shape (batch_size, seq_len, d_model), 
        # then out[:, -1] gives a tensor of shape (batch_size, d_model) corresponding to the last token's features for each batch sample. 
        
        # model.project is typically a linear layer (or projection layer) that maps the features 
        # from the model's hidden size (d_model) to the vocabulary size.
        # The output of model.project has shape (batch_size, vocab_size) and
        # represents the predicted probability distribution over the vocabulary for the next token.
        
        prob = model.project(out[: , -1 ])
        # we are fiding the max prob token using troch .max 
         
        
        _ , next_word = torch.max(prob , dim =1 )
        
        # we are concating the new token tot the input and sending it as the new inout 
        
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )
        
        if next_word == eos_idx:
            break
        
        
    return decoder_input.squeeze(0)



        
        
        
        
        
        
    


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    
    # here we are doing the validation so keep the mdoel in the eval  mdoe 
    model.eval()
    
    # set the required parameters 
    
    count =0 
    source_texts = []
    expected = []
    predicted = []
    
    #  this is just for making the output beautiful not for nay orhter thing 
    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80# 
        
    # for validation  remebr that that always write code of validation under torch.zero_grad 
    
    with torch.no_grad():
        for batch in validation_ds:
            
            count +=1 
            
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            
            #  check that the batchsize is 1 
            
            assert encoder_input.size(
               0 
            ) == 1 , " batch size must be one for validation "
            
            model_output = greedy_decode(model , encoder_input ,  encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device )
            
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())
            
            # here the line intends that it should not be added tot the computational grapg so we wre using the .detach()
            # and want to covert the gpu mode to cpu mode and tensor to numpy list 
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
            
            
    if writer : 
        # now lets add the parameters to the tensor board 
        # evaluate the charecter error rate  , word error rate , 
        # bleu metric 
        
        # -> char error rate :
        
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted , expected )
        writer.add_scalar('validation cer ' , cer , global_step )
        writer.flush()
        
        # compute the word error rate 
        
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted , expected )
        writer.add_scalar('validation wer ' , wer , global_step )
        writer.flush()
        
        
        # compute the blue metric 
        
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted , expected )
        writer.add_scalar('validation Bleu' , bleu , global_step )
        writer.flush()
        
        
# note for every big programs we see __name__ logic 
# here what it actually mean is :
    
    # normally 2 things happen 
    # 1) the python file directly running 
    # 2) it is being imported 
    
    # so inorder to differentiate it we use the __name__ 
    
    # if the file is being runned directly the  __name__ will have 
    # the value __main__ else it will have the name of the file 
    
    # so when we run the file directly the  code block under the __name__ 
    # will run else it will not run          


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
    
        
    
    
     

            
            
            
            
            
            
        
        
        
        
        
    
     
    
    
            
    

