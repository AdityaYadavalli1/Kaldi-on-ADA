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
- To run the recipe:
  - `cd egs/mini_librispeech/s5/` (asssuming you are the Kaldi directory already)
  - `./run.sh`
    - A good idea would be to use nohup. This makes sure that even if you lose your connection or if you logout, the program keeps running. And most times you want this to run in the background so that you can do something else meanwhile. For this, `nohup nice ./run.sh &`
- This recipe needs the package **flac**. Some of the nodes don't have it installed. But fortunately for me, after trying out a few nodes, I know for sure that nodes [13, 23, 15, 18, 30, 32, 34, 41] have it installed. I will add some more nodes to that list later on. Some nodes that don't have *flac* installed are [43, 46, 48, 50, 55, 57, 58, 48].
Note: Or you can create virtualenv and install it there.
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


- To **data/train_clean_5_sp/feats.scp** (my guess is extracted features are kept here)
    - Whenever you rerun run.sh (which you will. No one gets it right the first time), make sure that you delete this file. Otherwise the program stops after 1 hour and asks you delete. Even though the error message points exactly where the error is (so you can figure it out) it just wastes a lot of time so rather do this before you do `./run.sh`


- To **data/lang_chain**
  - Whenever you rerun run.sh, make sure that you delete this folder. Otherwise the program stops after 1-2 hour(s) and asks you delete this folder. Even though the error message points exactly where the error is (so you can figure it out) it just wastes a lot of time so rather do this before you do `./run.sh` 
