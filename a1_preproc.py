import sys
import argparse
import os
import json
import html
import re
import string
import spacy

indir = '/u/cs401/A1/data/';

# load spacys
nlp1002099953 = spacy.load('en', disable=['parser', 'ner'])

# load all abbrevations into a list
abbrev1002099953 = list(line1002099953.strip() for line1002099953 in open("/u/cs401/Wordlists/abbrev.english"))

# load all stopwords into a list
stop1002099953 = list(line1002099953.strip() for line1002099953 in open("/u/cs401/Wordlists/StopWords"))


def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''

    modComm = comment
    
    if 1 in steps:
        modComm = modComm.strip()
        modComm = modComm.replace('\n', ' ')
        modComm = modComm.strip()
        
    if 2 in steps:
        modComm = html.unescape(modComm)
        modComm = modComm.strip()
    
    if 3 in steps:
        # such regex detect tokends start with http, https or www
        modComm = re.sub(r"(http[^\s]*|www[^\s]*)\s*", "", modComm)
        # adjust sequential whitespace into one, for later simplicity
        modComm = re.sub(r"\s+", " ", modComm)
        modComm = modComm.strip()

    if 4 in steps:
        
        # split comment into word tokens by whitespace
        words = modComm.split(" ")
        
        # processing on each word token
        for word_idx in range(len(words)):
            next_word = words[word_idx]
            tracked_cmt = next_word
            track_idx = 0
            
            # processing on each character within word token
            for i in range(len(next_word)):
                char = next_word[i]
                if(char in string.punctuation):
                    
                    # apostrophes unchanged
                    if(char == '\''):
                        pass
                    # abbreviation unchanged
                    elif(next_word[:i+1] in abbrev1002099953):
                        pass
                    
                    # multiple punctuations case
                    # punctuation in middile position, not adding whitespace
                    elif((i != 0 and next_word[i-1] in string.punctuation) and (len(next_word) > i+1 and next_word[i+1] in string.punctuation)):
                        pass
                    # punctuation in leading position, add a whitespace at the front
                    elif((i != 0 and next_word[i-1] not in string.punctuation) and (len(next_word) > i+1 and next_word[i+1] in string.punctuation)):
                        tracked_cmt = tracked_cmt[:track_idx] + " " + tracked_cmt[track_idx:]
                        track_idx += 1
                    # punctuation in trailing position, add a whitespace at the end
                    elif(i != 0 and next_word[i-1] in string.punctuation):
                        tracked_cmt = tracked_cmt[:track_idx+1] + " " +tracked_cmt[track_idx+1:]
                        track_idx += 1
                    
                    # a stand alone punctuation
                    else:
                        tracked_cmt = tracked_cmt[:track_idx] + " " + tracked_cmt[track_idx] + " " + tracked_cmt[track_idx+1:]
                        track_idx += 2
                
                # update processed word token
                words[word_idx] = tracked_cmt.strip()
                track_idx += 1
        
        # rejoin comment
        modComm = " ".join(words)
        modComm = modComm.strip()
    
    if 5 in steps:
        
        # split comment into word tokens by whitespace
        words = modComm.split(" ")
        
        # processing on each word token
        for word_idx in range(len(words)):
            next_word = words[word_idx]
            
            # processing on each character of word token
            for i in range(len(next_word)):
                char = next_word[i]
                
                # clitics
                if(char == '\''):
                    if(i+1 < len(next_word)):
                        # n't case
                        if(next_word[i-1] == 'n' and next_word[i+1] == 't'):
                            words[word_idx] = next_word[:i-1] + " " + next_word[i-1:]
                        # 've, 'll etc
                        else:
                            words[word_idx] = next_word[:i] + " " + next_word[i:]
                    # s' etc
                    else:
                        words[word_idx] = next_word[:i] + " " + next_word[i:]

        # rejoin comment
        modComm = " ".join(words)
        modComm = modComm.strip()
        
    if 6 in steps:
        
        # load model
        utt = nlp1002099953(modComm)
        
        # add tag
        modComm = ""
        for token in utt:
            modComm = modComm + token.text + '/' + token.tag_ + " "
        
        modComm = modComm.strip()
    
    if 7 in steps:    
        
        # split into word tokens by whitespace
        words = modComm.split(" ")
        
        # check if stopwords for each word token
        modComm = ""
        for next_word in words:
            slash = next_word.rfind("/")
            word = next_word[:slash]
            if(word not in stop1002099953):
                modComm = modComm + next_word + " "
        
        modComm = modComm.strip()
        
    if 8 in steps:
        
        # split into word tokens
        words = modComm.split(" ")
        
        # retrieve the raw comment
        raw_com = ""
        for next_word in words:
            slash = next_word.rfind("/")
            # extract the raw word exclude tag
            word = next_word[:slash]
            raw_com = raw_com + word + " "
        
        # apply lemmatization
        utt = nlp1002099953(raw_com)
        modComm = ""
        for token in utt:

            # keep the original token
            if(token.lemma_[0] == '-' and token.text[0] != '-'):
                modComm = modComm + token.text + "/" + token.tag_ + " "
            # otherwise, change token to lemma
            else:
                modComm = modComm + token.lemma_ + "/" + token.tag_ + " "

        modComm = modComm.strip()
            
    if 9 in steps:
        
        # split word into word tokens
        words = modComm.split(" ")
        
        # process each word token
        modComm = ""
        for i in range(len(words)):
            curr_word = words[i]
            slash = curr_word.rfind("/")
            word = curr_word[:slash]
            tag = curr_word[slash:]
                        
            # end of punctuation tag is "/." which includes ".", "?", "!", "..", "..."
            # NFP tag is for "...." or longer
            if(tag == "/." or tag == "/NFP"):
                
                # if such punctuation belongs to abbrevation
                is_abbrev = False
                if(i != 0):
                    prev_word = words[i-1]
                    prev_slash = prev_word.rfind("/")
                    prev_word_only = prev_word[:prev_slash]
                    concat = prev_word_only + word
                    is_abbrev = (concat in abbrev1002099953)
                
                # if such punctuation belongs to a consecutive punctuations
                next_punc = False
                if(i+1 < len(words)):
                    next_word = words[i+1]
                    next_slash = next_word.rfind("/")
                    next_tag = next_word[next_slash:]
                    next_punc = (next_tag == "/.")
                
                # do not add new line for abbreviation (best judgement)
                if(is_abbrev):
                    modComm = modComm + curr_word + " "
                    pass
                # do not add new line for punctuation in the middle of several punctuations
                elif(next_punc):
                    modComm = modComm + curr_word + " "
                    pass
                # otherwise a valid end of sentence
                else:
                    modComm = modComm + curr_word + "\n"
            
            else:
                modComm = modComm + curr_word + " "
        modComm = modComm.strip()
    
    if 10 in steps:
        
        # split word into word tokens
        lines = modComm.split("\n")
        
        modComm = ""
        # process each sentences
        for line in lines:
            # split word into word tokens
            words = line.split(" ")

            # process each word token
            for i in range(len(words)):
                curr_word = words[i]
                slash = curr_word.rfind("/")
                word = curr_word[:slash]
                tag = curr_word[slash:]

                # lower case the word, tag remains the same
                word = word.lower()
                modComm = modComm + word + tag + " "
            
            modComm.strip()
            modComm = modComm + "\n"
            
        modComm = modComm.strip()
        
    return modComm


def main( args ):

    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))
            
            ##################################
            # select appropriate args.max lines
            i = args.ID[0]%len(data)
            counter = 0
            # read those lines with something like `j = json.loads(line)`
            while counter < int(args.max):
                # wrap around
                if(i == len(data)):
                    i = 0
                
                # for comfort during process on full data :D
                print(counter + 1)
                
                line = data[i]
                j = json.loads(line)
                # choose to retain fields from those lines that are relevant to you
                curr_line = {}
                curr_line['id'] = j['id']
                curr_line['body'] = j['body']
                # add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 
                curr_line['cat'] = fullFile.split('/')[-1]
                # process the body field (j['body']) with preproc1(...) using default for `steps` argument
                temp = preproc1(j['body'])
                # replace the 'body' field with the processed text
                curr_line['body'] = temp
                # append the result to 'allOutput'
                allOutput.append(curr_line)

                i += 1
                counter += 1
    
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (int(args.max) > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
        
    main(args)

