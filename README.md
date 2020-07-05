# Kaldi on ADA

[WED-10th June '20]
## Installing Kaldi
1. Go to the directory where you want to place the Kaldi code.
2. Run `git clone https://github.com/kaldi-asr/kaldi.git kaldi --origin upstream`
3. To get into kaldi directory `cd kaldi`
4. Following the instructions given in INSTALL:
  1. `cd tools`
  2. To check if all the dependencies are installed, run: `./extras/check_dependencies.sh`. You will get a message **"All Okay"** when all the dependencies are installed.
    - Note: It is recommended to use g++ version 4.8. When you use other versions you might run into some errors such as not installing/compiling OpenFST (I ran into this). Luckily, on ADA the default g++ version is 4.8.  
  3. Then do `make -j X` where X is the number of CPUs you request.
  4. `cd ../src`
  5. `./configure --shared`
  6. `make depend -j X` where is X is the number of CPUs you request.
  7. `make -j X` where is X is the number of CPUs you request.

With this you must have Kaldi installed successfully on your ADA.

To test your installation:
1. `cd egs/yesno/s5`
2. `./run.sh`

Since its a very simple corpus you should get 0 WER. Something or this sort: **%WER 0.00 [ 0 / 232, 0 ins, 0 del, 0 sub ] exp/mono0a/decode_test_yesno/wer_10_0.0**. Any other result means that you have not installed it correctly. Either redo it or follow the error messages carefully.

[SAT-13th June '20]
## Trying out Kaldi Recipes
### Mini Librispeech
#### Running the recipe successfully
- To run the recipe:
  - `cd egs/mini_librispeech/s5/` (asssuming you are the Kaldi directory already)
  - `./run.sh`
    - A good idea would be to use nohup. This makes sure that even if you lose your connection or if you logout, the program keeps running. And most times you want this to run in the background so that you can do something else meanwhile. For this, `nohup nice ./run.sh &`
- This recipe needs the package **flac**. Some of the nodes don't have it installed. But fortunately for me, after trying out a few nodes, I know for sure that nodes [13, 23, 15, 18, 30, 31 32, 34, 40, 41] have it installed. I will add some more nodes to that list later on. Some nodes that don't have *flac* installed are [43, 46, 48, 50, 55, 57, 58, 48].
- To request a specific node (although might take some more time than usual for it to get allocated to you): `sinteractive -c X -g Y -w gnodeXX` where X is the number of CPUs you want to request, Y is the number of GPUs you want to request and XX is the gnode number you want.

- Make sure that you request around 40 CPUs (maybe -10) and 3 GPUs (maybe -2) so that everything runs as intended.
  - sinteractive -c 40 -p long -A research -g 3

[SUN-14th June '20]\
Made some changes to code so that it runs on ADA
- To **cmd.sh**
    - Commented all the 3 lines of code.
    - Added these 3 lines:

    ```
      export train_cmd=run.pl
      export decode_cmd=run.pl
      export mkgraph_cmd="run.pl"
    ```
- To **local/chain2/tuning/run_tdnn_1a.sh**
  - I got $cuda_cmd" is unbounded variable error. It pointed at line 297 in the file. So I changed "$cuda_cmd" to "$train_cmd". This might be a bug in Kaldi because I've seen a commit where all cuda_cmds were changed to train_cmds in another recipe. (or maybe I have to load cuda module. I have to check this). I did not get this error after changing this line though.


- To **data/train_clean_5_sp/feats.scp** (extracted features)
    - Whenever you rerun run.sh (which you will. No one gets it right the first time), make sure that you delete this file. Otherwise the program stops after 1 hour and asks you delete. Even though the error message points exactly where the error is (so you can figure it out) it just wastes a lot of time so rather do this before you do `./run.sh`


- To **data/lang_chain**
  - Whenever you rerun run.sh, make sure that you delete this folder. Otherwise the program stops after 1-2 hour(s) and asks you delete this folders. Even though the error message points exactly where the error is (so you can figure it out) it just wastes a lot of time so rather do this before you do `./run.sh`

[MON 15th Jun '20]
- Deleted all the folders that weren't there before I ran `./ran.sh` for the first time and ran `./ran.sh` again. The deleted folders are:
  - data
  - exp
  - corpus
  - mfcc
- Got an error in **steps/chain2/train.sh**
 - **run.sh** calls **local/chain2/run_tdnn.sh** (default model) which in turn calls **steps/chain2/train.sh**.
 - [ERROR] steps/chain2/train.sh: error detected training on iteration 1
 - The logs (for me it at **exp/chaina/tdnn2c_sp/log/train.1.3.log**) reads out that GPUs should be in compute-exclusive mode. To set it to compute exclusive mode (I didn't try this yet but the log suggests) **Suggestion: use 'nvidia-smi -c 3' to set compute exclusive mode**. Not setting GPUs to compute exclusive mode led to this: **Failed to allocate a memory region of 622854144 bytes.  Possibly this is due to sharing the GPU.  Try switching the GPUs to exclusive mode (nvidia-smi -c 3) and using the option --use-gpu=wait to scripts like steps/nnet3/chain/train.py.  Memory info: free:1187M, used:9991M, total:11178M, free/total:0.106186 CUDA error: 'out of memory'**
 - Things I have tried to get rid of the error:
     - Tried the recommended command: `nvidia-smi -c 3`. I didn't have the permissions to run it.


[TUE 16th Jun '20]
- Tried to run it without GPU to see whether it can run without it. The idea was if it did then since the error was coming from CUDA. I can bypass it. That didn't work.

[THURS 18th Jun '20]\
In **local/chain2/train.sh**
- Changed number of jobs (both initial and final to 1) and added use-gpu = wait flag.
  - Note: Make sure your node has flac installed or else you will get some more errors.
- After making that change, I got **run.pl: 75 / 75 failed, log is in exp/chaina/tdnn2c_sp/raw_egs/log/get_egs.*.log** error when I ran `./local/chain2/run_tdnn.sh`\
[FRI 19th Jun '20]
  - Note: With these changes, I was able to run it on the SPL server. But at this point, for ADA, it seems as though we need to make some more changes to Kaldi scripts.

[SAT 20th Jun '20]
#### Understanding the recipe
I will be going through `run.sh`. The top level script that calls other scripts.
- Before Stage 0:
The script `local/download_and_untar.sh` downloads the data you want and unzips it. First argument is the should tell the program where the user wants to download the data, second is url and the third is corpus part. This will any way be shown if you provide incorrect arguments.
- **Stage 0:**
This downloads 3 language models (why three? I don't know). First argument to the script called (`local/download_lm.sh`) is the base url and the second argument is the where those models are stored (folder).

  [THURS 25th Jun '20]
- **Stage 1:**
In this stage, 5 scripts are called:
  - `local/data_prep.sh`:
  Creates all the important files (wav.scp, transcript, utt2spk, spk2gender, utt2dur) given that the data has SPEAKERS.txt file. This takes source directory as the first argument and destination directory as the second argument.
  - `local/prepare_dict.sh`:
  Creates all the necessary steps for g2p. If lexicon_nosil.txt is not available, using CMUdict g2p (and Sequitur G2P*) lexicon_nosil.txt. With this all g2p files are created (lexicon.txt, silence.txt, nonsilence.txt, optional.txt, extraquestions.txt) . Arguments need:
   - 1st argument: LM directory (in our case `data/local/lm`)
   - 2nd argument: G2P model directory (it was not used in our case because we started from stage 3)
   - 3rd argument: destination directory (where all the files made are stored)
  - `utils/prepare_lang.sh`:
    - 1st argument: Source directory (the destination directory for `local/prepare_dict.sh`)
    - 2nd argument: OOV word representation (eg: "UNK")
    - 3rd argument: Temporary directory (this script needs a space to store some temporary files)
    - 4th argument: Destination directory

   With these, essentially, it will make **L.fst** (lexicon in fst format. The format used by kaldi to calculate scores) [all of this is usually stored in `data/lang`]
  - `local/format_lms.sh`:
    - optional argument: `utils/prepare_lang.sh` destination directory should be source directory for this file (by default it is `data/lang`)
    - 2nd argument: LM directory (`data/local/lm` in this case)

    This script converts arpa formatted language model to **G.fst** (grammer in fst format). Separate fst models (L and G) are avaiable in `${src_dir}_test_${lm_suffix}` where lm suffix could something of the form of tgsmall or tgmed or tglarge.
  - `utils/build_const_arpa_lm.sh`: Converts arpa formatted language model to const arpa format. Arpa is the format a language model is in and constant arpa is the format that is needed for decoding (in `run.sh` this is run only on tglarge. We can run the previous script on tglarge too but that'll be too time consuming)

  [FRI 26th Jun '20]
- **Stage 2:**
In stage 2 we extract mfcc features from wav files. For this, 3 scripts are called:
  - `steps/make_mfcc.sh`:
    - 1st argument: Directory where the data is stored. (training)
    - 2nd argument: Where you want want the log files to be written. (optional argument)
    - 3rd argument: Where you want the extracted mfcc feats to be written. (optional argument)

    This extracts mfcc features from the wav files.   
  - `steps/compute_cmvn_stats.sh`:
    - 1st argument: Training data directory
    - 2nd argument: Log directory
    - 3rd argument: Result files' directory
    Compute cepstral mean and variance statistics per speaker. (probably for speaker normalisation). MFCC features are stored in \*.ark files and the absolute filepaths are stored in \*.scp inside the 3rd argument (directory). For a more in depth explanation (I do not need it and for lack better documentation skills) you can refer [this](http://jrmeyer.github.io/misc/kaldi-documentation/kaldi-documentation.pdf)
  - `utils/subset_data_dir.sh`: A nice utility script. Given a data file with a specified number of utterances (a subset of utterances), it outputs a subset of utterances from the data file. You can shortest, last or first n utterances from the file by specifying it with the corresponding argument.
    - 1st argument: Option ([shortest, last, first n] utterances)
    - 2nd argument: Data
    - 3rd argument: Resultant data

- **Stage 3:**
In stage 3, we train a monophone system. For this, 3 scripts are called:
  [SUN 5th Jul'20]
  - `steps/train_mono.sh`:
      - 1st argument: Training data directory
      - 2nd argument: lang directory (eg: `data/lang_nosp` or `data/lang`)
      - 3rd argument: A place where the model can be stored (`exp/mono`). Apparently this can take a few gigabytes.
      Scripts it calls:
        - `gmm init mono` with training features extracted in the previous stage. This creates a phonetic decision tree with only the root because all GMMs are initialised to the same value. One can see the topology of HMMs in `data/lang/topo`.
        - `compile-train-graphs` which generates a training graph - one FST per training utterance. These FSTs encode HMM structure for that training utterance. The FSTs' input-symbols are transitions-ids which includes pdf-ids (which represents GMM acoustic state) [transition ids essentially encode audio frame] and output-symbols are words. This FST also includes cost (mostly including the cost coming from lexicon i.e pronunciation probability) but the transition probability of the HMM model will only be added later during training (word-word or phone-phone cost is added later)
        - `align-equal-compiled` this performs the most naïve yet the best the guess when we don't know anything - equally spaced alignments. That is this assumes that all HMM states are equally spaced.
        - `gmm-est` force aligns using Viterbi Training. Using this, GMMs are re-estimated.
        Inside the training loop the following scripts are called: [basically EM algorithm is applied. Baum-Welch in a sense]
        - `gmm-align-compiled` aligns phone states according to the GMM models.
        - `gmm-acc-stats-ali` accumulate stats for GMM training.
        - `gmm-est` performs Maximum Likelihood to re-estimate the GMM-based acoustic models. (This time with different options)
        
        **NOTE:** In this script we can change the beam search size. It is a hyper-parameter, change this from data to data to get optimal results.\
        **NOTE1:** For any graph, the output symbol is words (specifically `word.txt`) and inputs are transition-ids (arcs in CD HMMs) (as opposed to pdf-ids which represent GMM states)

  - `utils/mkgraph.sh`: This script creates a fully expanded decoding graph (HoCoLoG) that represents all the language-model, pronunciation dictionary (lexicon), context-dependency, and HMM structure in our model. The output is a Finite State Transducer that has word-ids on the output, and pdf-ids on the input (these are indexes that resolve to Gaussian Mixture Models).
    - 1st argument: lang directory
    - 2nd argument: directory where the model is stored (`exp/mono`)
    - 3rd argument: directory where the decoded graph can be stored (`exp/mono/graph_nosp`) [graph generation]
  - `steps/decode.sh`: This script finally decodes the graph compiled using `mkgraph.sh` i.e we generate lattices (a graph-based record of the most likely utterances) using scores made by `local/score.sh`

  **Note:** “decoding” refers to the computation where we find the best sentence given the model.
  - `steps/align_si.sh`: It combines all the alignments learnt in different passes/epochs by `gmm-est` and `gmm-align-compiled`.





## Some Useful Sources
- ASR modelling with mini_librispeech recipe in Kaldi
  - [Explanation of basics of ASR modelling with HMMs. Part 1](https://medium.com/@qianhwan/understanding-kaldi-recipes-with-mini-librispeech-example-part-1-hmm-models-472a7f4a0488)
  - [Explanation of DNN acoustic modelling. Part 2](https://medium.com/@qianhwan/understanding-kaldi-recipes-with-mini-librispeech-example-part-2-dnn-models-d1b851a56c49)
- [Understanding Example scripts with RM speech corpus](https://kaldi-asr.org/doc/tutorial_running.html)
- [Step by Step guide for Acoustic Modelling with Kaldi](https://eleanorchodroff.com/tutorial/kaldi/training-acoustic-models.html)
- [Josh's Kaldi Notes. Good for basic theory](http://jrmeyer.github.io/asr/2016/02/01/Kaldi-notes.html)
- [Step by Step guide for Modelling ASR with Kaldi. Might not be upto date](http://white.ucc.asn.au/Kaldi-Notes/install_notes)
- [Malayalam digit recogniser recipe. Could be useful if I want to make a recipe later on](https://github.com/kavyamanohar/malayalam-spoken-digit-recognizer)
- [Some theory on probabilitistic graph formuation and solving. Useful for decoding](https://drive.google.com/drive/folders/1wmncLdRsY27Av1oNzk7Jaa9xWhxGTPrQ)
- [Best Series of articles that explain ASR theory](https://medium.com/@jonathan_hui/speech-recognition-series-71fd6784551a)
- [Povey's lectures. Old and not upto date but very well explained](http://www.danielpovey.com/kaldi-lectures.html)
- Kaldi Forums are pretty active
