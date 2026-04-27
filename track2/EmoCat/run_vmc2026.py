## Modified from the code example here:
## https://github.com/ddlBoJack/emotion2vec

indir = '/PATH/TO/YOUR/wavdir'  ### EDIT THIS

'''
Using the finetuned emotion recognization model

rec_result contains {'feats', 'labels', 'scores'}
        extract_embedding=False: 9-class emotions with scores
        extract_embedding=True: 9-class emotions with scores, along with features

9-class emotions:
iic/emotion2vec_plus_seed, iic/emotion2vec_plus_base, iic/emotion2vec_plus_large (May. 2024 release)
iic/emotion2vec_base_finetuned (Jan. 2024 release)
    0: angry
    1: disgusted
    2: fearful
    3: happy
    4: neutral
    5: other
    6: sad
    7: surprised
    8: unknown
'''

from funasr import AutoModel
import os

model_id = "iic/emotion2vec_plus_large"
outf = open(model_id.split('/')[1] + '.out', 'w')

model = AutoModel(
    model=model_id,
    hub="hf",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
)

outf = open('category_probs.out', 'w')

wavs = os.listdir(indir)
for w in wavs:
    wav_file = indir + '/' + w
    rec_result = model.generate(wav_file, output_dir="./outputs", granularity="utterance", extract_embedding=False)
    labels = rec_result[0]['labels']
    scores = rec_result[0]['scores']
    outl = w + '|'
    for i in range(len(labels)):
        l = labels[i]
        if len(l.split('/')) > 1:
            l = l.split('/')[1]
        if l not in ['angry', 'happy', 'neutral', 'sad', 'surprised']:
            continue
        s = str(scores[i])
        outl += l + ':' + s + '|'
        outl += '\n'
        outf.write(outl)
    
