import numpy as np
import sys
import argparse
import os
import json

import string
import csv

# feature categories file
fp1002099953 = ["I", "me", "my", "mine", "we", "us", "our", "ours"]
sp1002099953 = ["you", "your", "yours", "u", "ur", "urs"]
tp1002099953 = ["he", "him", "his", "she", "her", "hers", "it", "its", "they", 
                "them", "their", "theirs"]
slang1002099953 = ["smh", "fwb", "lmfao", "lmao", "lms", "tbh", "rofl", "wtf", 
                   "bff", "wyd", "lylc", "brb", "atm", "imao", "sml", "btw", 
                   "bw", "imho", "fyi", "ppl", "sob", "ttyl", "imo", "ltr",
                   "thx", "kk", "omg", "omfg", "ttys", "afn", "bbs", "cya", "ez", 
                   "f2f", "gtr", "ic", "jk", "k", "ly", "ya", "nm", "np", "plz", 
                   "ru", "so", "tc", "tmi", "ym", "ur", "u", "sol", "fml"]

# feats file
left_feats1002099953 = np.load("/u/cs401/A1/feats/Left_feats.dat.npy")
center_feats1002099953 = np.load("/u/cs401/A1/feats/Center_feats.dat.npy")
right_feats1002099953 = np.load("/u/cs401/A1/feats/Right_feats.dat.npy")
alt_feats1002099953 = np.load("/u/cs401/A1/feats/Alt_feats.dat.npy")

# categories id mapping
left_ids1002099953 = list(id1002099953.strip() for id1002099953 in open("/u/cs401/A1/feats/Left_IDs.txt"))
center_ids1002099953 = list(id1002099953.strip() for id1002099953 in open("/u/cs401/A1/feats/Center_IDs.txt"))
right_ids1002099953 = list(id1002099953.strip() for id1002099953 in open("/u/cs401/A1/feats/Right_IDs.txt"))
alt_ids1002099953 = list(id1002099953.strip() for id1002099953 in open("/u/cs401/A1/feats/Alt_IDs.txt"))

# read in norms into dictionary, key is word, content is a list of information
bg1002099953 = {}
with open("/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv") as csvfile1002099953:
    csv_reader1002099953 = csv.reader(csvfile1002099953)
    # flag for skipping the first element (header)
    flag1002099953 = 0
    for line1002099953 in csv_reader1002099953:
        word1002099953 = line1002099953[1]
        if(word1002099953 != "" and flag1002099953 != 0):
            # extract content of [aoa, img, fam]
            bg1002099953[word1002099953] = [float(line1002099953[3]), float(line1002099953[4]), float(line1002099953[5])]
        flag1002099953 = 1

war1002099953 = {}
with open("/u/cs401/Wordlists/Ratings_Warriner_et_al.csv") as csvfile1002099953:
    csv_reader1002099953 = csv.reader(csvfile1002099953)
    # flag for skipping the first element (header)
    flag1002099953 = 0
    for line1002099953 in csv_reader1002099953:
        word1002099953 = line1002099953[1]
        if(word1002099953 != "" and flag1002099953 != 0):
            # extract content of [vms, ams, dms]
            war1002099953[word1002099953] = [float(line1002099953[2]), float(line1002099953[5]), float(line1002099953[8])]
        flag1002099953 = 1


def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    
    # initialize
    feats = np.zeros(173)
        
    # parse input
    comment_copy = comment
    comment_copy = comment_copy.replace("\n", " ")
    cmts = comment_copy.split(" ")
    
    # initialize some feature count
    fp_count = 0
    sp_count = 0
    tp_count = 0
    conj_count = 0
    past_verb = 0
    future_verb = 0
    comma_count = 0
    punc_count = 0
    common_noun = 0
    proper_noun = 0
    adverb = 0
    wh = 0
    slang = 0
    upper = 0
    aoas = []
    imgs = []
    fams = []
    vmss = []
    amss = []
    dmss = []
    
    # process on each comments
    for i in range(len(cmts)):
        
        # split current comment into word and tag
        cmt = cmts[i]
        cmt_idx = cmt.rfind("/")
        word = cmt[:cmt_idx]
        tag = cmt[cmt_idx+1:]
        
        # feature 1: first person pronoun
        if(word in fp1002099953):
            fp_count += 1
        # feature 2: second person pronoun
        if(word in sp1002099953):
            sp_count += 1
        # feature 3: third person pronoun
        if(word in tp1002099953):
            tp_count += 1
            
        # feature 4: coordinating conjunction
        if(tag == 'CC'):
            conj_count += 1
        # feature 5: past tense verb
        if(tag == 'VBD'):
            past_verb += 1
        
        # feature 6: future tense verb
        if(word == "will" or word == "gonna" or "'ll" in word):
            future_verb += 1  
        if(word == "going"):
            # checking following is to+VB
            if(i+2 < len(cmts)):
                cmt1 = cmts[i+1]
                cmt2 = cmts[i+2]
                idx1 = cmt1.rfind("/")
                idx2 = cmt2.rfind("/")
                word1 = cmt1[:idx1]
                tag2 = cmt2[idx2+1:]
                if(word1 == "to" and tag2 == "VB"):
                    future_verb += 1
        
        # feature 7: commas
        if(tag == ","):
            comma_count += 1
        # feature 8: multichar punctuation tokens
        if(len(word) >= 2):
            punc = 0
            for next_char in word:
                if(next_char in string.punctuation):
                    punc += 1
            if(punc == len(word)):
                punc_count += 1
                
        # feature 9: common nouns
        if(tag == "NN" or tag == "NNS"):
            common_noun += 1
        # feature 10: proper nouns
        if(tag == "NNP" or tag == "NNPS"):
            proper_noun += 1
        # feature 11: adverbs
        if(tag == "RB" or tag == "RBR" or tag == "RBS"):
            adverb += 1
        # feature 12: wh- words
        if(tag == "WDT" or tag == "WP" or tag == "WP$" or tag == "WRB"):
            wh += 1
        # feature 13: slang
        if(word in slang1002099953):
            slang += 1
            
        # feature 14: upper case
        if(len(word) >= 3 and word.isupper()):
            upper += 1
            
        # feature 18 - 23: norms
        if(word in bg1002099953.keys()):
            bg = bg1002099953[word]
            aoas.append(bg[0])
            imgs.append(bg[1])
            fams.append(bg[2])
            
        if(word in war1002099953.keys()):
            war = war1002099953[word]
            vmss.append(war[0])
            amss.append(war[1])
            dmss.append(war[2])
    
    # extract feature 17: number of sentences
    sentences = comment.splitlines()
    num_sentences = len(sentences)
    
    # number of tokens exclude punctuation-only ones
    token_count = 0
    # total length of tokens exclude punctuation-only, in character
    token_len = 0
    # number of tokens include everything
    sent_tokens = 0
    for next_sent in sentences:
        tokens = next_sent.split(" ")
        sent_tokens += len(tokens)
        for next_token in tokens:
            slash = next_token.rfind("/")
            word = next_token[:slash]
            # not punctuation-only
            if(not (len(word) == 1 and word in string.punctuation)):
                token_count += 1
                token_len += len(word)
    
    # feature 15: avg length of sentences, in tokens
    if(num_sentences == 0):
        avg_sent = 0
    else:
        avg_sent = sent_tokens / num_sentences
    
    # extract feature 16: avg length of tokens, in characters
    if(token_count == 0):
        avg_token = 0
    else:
        avg_token = token_len / token_count
    
    # extract feature 18 - 20
    avg_aoa = np.mean(aoas) if not str(np.mean(aoas)) == "nan" else 0
    avg_img = np.mean(imgs) if not str(np.mean(imgs)) == "nan" else 0
    avg_fam = np.mean(fams) if not str(np.mean(fams)) == "nan" else 0
    
    # extract feature 21 - 23
    sd_aoa = np.std(aoas) if not str(np.std(aoas)) == "nan" else 0
    sd_img = np.std(imgs) if not str(np.std(imgs)) == "nan" else 0
    sd_fam = np.std(fams) if not str(np.std(fams)) == "nan" else 0
    
    # extract feature 24 - 26
    avg_vms = np.mean(vmss) if not str(np.mean(vmss)) == "nan" else 0
    avg_ams = np.mean(amss) if not str(np.mean(amss)) == "nan" else 0
    avg_dms = np.mean(dmss) if not str(np.mean(dmss)) == "nan" else 0
    
    # extract feature 27 - 29
    sd_vms = np.std(vmss) if not str(np.std(vmss)) == "nan" else 0
    sd_ams = np.std(amss) if not str(np.std(amss)) == "nan" else 0
    sd_dms = np.std(dmss) if not str(np.std(dmss)) == "nan" else 0
    
    # feed into feats
    feats[0] = fp_count
    feats[1] = sp_count
    feats[2] = tp_count
    feats[3] = conj_count
    feats[4] = past_verb
    feats[5] = future_verb
    feats[6] = comma_count
    feats[7] = punc_count
    feats[8] = common_noun
    feats[9] = proper_noun
    feats[10] = adverb
    feats[11] = wh
    feats[12] = slang
    feats[13] = upper
    feats[14] = avg_sent
    feats[15] = avg_token
    feats[16] = num_sentences
    feats[17] = avg_aoa
    feats[18] = avg_img
    feats[19] = avg_fam
    feats[20] = sd_aoa
    feats[21] = sd_img
    feats[22] = sd_fam
    feats[23] = avg_vms
    feats[24] = avg_ams
    feats[25] = avg_dms
    feats[26] = sd_vms
    feats[27] = sd_ams
    feats[28] = sd_dms
    
    return feats
    

def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))

    # each comment
    for i in range(len(data)):
        
        # for comfort during processing :D
        print(i + 1)
        
        # parse json
        curr_line = data[i]
        curr_cmt = curr_line['body']
        curr_id = curr_line['id']
        curr_cat = curr_line['cat']
        
        # extract the first 29 features
        feats[i, :173] = extract1(curr_cmt)

        # fill in 30 - 173 features, and labels, by read in feats dictionareis generated
        if(curr_cat == "Left"):
            feats[i, 29:173] = left_feats1002099953[left_ids1002099953.index(curr_id)]
            feats[i, 173] = 0
        elif(curr_cat == "Center"):
            feats[i, 29:173] = center_feats1002099953[center_ids1002099953.index(curr_id)]
            feats[i, 173] = 1
        elif(curr_cat == "Right"):
            feats[i, 29:173] = right_feats1002099953[right_ids1002099953.index(curr_id)]
            feats[i, 173] = 2
        elif(curr_cat == "Alt"):
            feats[i, 29:173] = alt_feats1002099953[alt_ids1002099953.index(curr_id)]
            feats[i, 173] = 3
            
    np.savez_compressed( args.output, feats)
    
    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
    
    main(args)
    
