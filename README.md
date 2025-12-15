 











CPSC 483 - Machine Learning
California State University, Fullerton
Instructor: Tseng-Ching Shen

Project Title: Voice Cloner

Group Members:
Jacob Nguyen
Anderson Pham
			


Functionalities

	Overview:
The idea we had for our voice cloner was to run everything locally and have fast voice cloning capabilities. To do this, we would need to utilize “Zero Shot Voice Cloning.” This technology essentially creates embeddings of a 5-15 second voice clip, and uses it inside the model through one forward pass. Instead of fine-tuning, which takes several hours to days, the one pass made by zero-shot voice cloning takes only a few seconds to a few minutes. Given the speed advantages, zero-shot voice cloning was chosen as the main technology for this project. To save time on user training, the best approach was to have the user download a model that we trained separately (or train a model with our custom scripts) and use that with our program files.

Using Zero-Shot Voice Cloning:
The user can:
Select which model to use if it is in the “models” folder in the root directory of the program.
Input the location of the .txt transcript and .wav for zero-shot voice cloning.
Type or select a txt file to be run through the model using the cloned voice.
Do zero-shot voice cloning on either CPU or GPU (GPU with CUDA is heavily preferred).
Decide what to save their voice clip as.
Change the temperature (makes it more monotone or expressive).

Once the user selects all their settings, the program will perform zero-shot voice cloning on the specified text using their chosen voice. 
	

Implementation Approach

Current:
To fine tune a model, the F5-TTS tool kit was used. First, we needed to select, clean, and format a dataset. In order to get better zero-shot voice cloning capabilities, we decided that getting multiple speakers from “asmr roleplay videos” was the best way for us to get clean audio and variety. Most datasets from online were already used to train the pretrained model, thus we had to create our own dataset. First, we needed to download the data and change the sampling rate to be 24kHz. We also needed to divide each sentence into its own voice clip, and create a transcription for each one. The data was then ported over to a csv, which the F5-TTS toolkit used to interpret our data. Once the model was trained, we were able to use it with the same config.json and vocab.txt files that the pretrained model we used as the base for fine tuning. 

We utilize the f5-tts library (must be downloaded through pip) in our code to take in our parameters, pass them into f5-tts, and run several functions to perform zero shot voice cloning internally. 

Our code and f5-tts:

	Zero-Shot-NHP.py
This is the file the user runs. It is a CLI command interface, where the user must pick where input audio and transcriptions, and text to be translated are located, the model to use, the temperature and speed of the final product, the name of the output file, and if the training computer should use CUDA GPU power or CPU power to do zero shot voice cloning.

model.py
This file defines the core neural network architecture used by F5 TTS.  This model is a diffusion transformer trained using conditional flow matching and uses mel spectrograms.  It takes noisy mel spectrogram inputs along with conditioning information derived from reference audio and text then embeds time steps for diffusion and iteratively denoises the signal through a stack of transformer blocks.  This file does not handle preprocessing or inference logic

utils.py
This file is a shared module that assists with data preprocessing, reproducibility, and sampling efficiency across the F5-TTS system

utils_infer.py
This file handles a bulk of the inference pipeline.  It takes in audio, cleans it, and then performs conditional flow-matching diffusion to generate mel spectrograms which implicitly captures the voice’s identity.  Then it decodes those spectrograms into waveforms and creates synthesized speech.  This is also where the vocoder logic is taking place, the library uses BigVGAN Nvidia to implement it.

cfm.py
This file provides the core model of Conditional Flow Matching, used for speech generation and editing. It wraps a transformer to learn and sample continuous flows between noise and target mel spectrograms conditioned on reference audio and text.


Discussion about your results

The fine tuned version sounds significantly cleaner because it has less static and sounds more fluid and natural. The difference is most noticeable with headphones. The audio also sounds like the speaker is closer to the microphone. We thought the end voice would be more whispery, but it was not because of the type of audio we choose to make our dataset from. The small dataset was made with asmr “roleplay” clips, which are more like a conversation from your perspective, rather than traditional asmr videos with lip smacking and weird sounds playing. The training was done over the cloud with google colab. We loaded the dataset into a google drive folder, and mounted the drive into the storage in the colab environment. Once we downloaded the necessary dependencies, f5-tts ran the fine tuning for 5 epochs. We had access to a t4 GPU through google cloud, which let us use a batch size of 4. This ran for a total of 8 hours, and completed 2090 steps. The reason we chose only 5 epochs was because we did not want to risk overfitting and time constraints of the project. Overall, the fine tuning process took a long time to get started, but worked much better than we could have anticipated, and was worth it in the end.

Training was by far the hardest part about this project. Training from scratch was incredibly difficult due to errors, large file sizes and not enough storage, file corruption, and issues in real life that led to the project difficulties. Hardware difficulties were also a large concern, as our 4060 GPU was not powerful enough to run training using the gradio application included inside of F5-TTS. The hardware issue was solved with cloud computing, but the process was annoying and difficult due to the program randomly being stopped by google time limits. Organizing and cleaning the data took a very long time to do. Tools like “whisper” made the process very easy, but many times, the tools did not want to accept our formatted and organized datasets.

Legacy/old non working components:
Our program did not start off as what the final product is now at all, and MANY changes were made throughout its life span. Originally, we were trying to train a model from scratch with XTTS using “Coqui,” but due to endless bugs and failed troubleshooting, we gave up and switched our approach and used “F5TTS” instead. We attempted to implement the code with fewer tools and more basic libraries to better convey the complexity of a voice cloner. Due to time constraints, we were unable to get working scripts to train a model from scratch using the “train_clean_100” dataset from LibriTTS, and VCTK Corpus dataset. This was done using the train_fast.py script (not in the repo), which let the user train a new model from scratch. This must be done as the architecture for the legacy program’s zero-shot voice cloning requires specific architecture in order for functionality. We also trained other types of models from scratch with other tools such as coqui. Originally, we trained a model for 240 epochs. This took 3 days of straight training, and the results did not sound good, and could not use zero shot voice cloning. We gathered from these experiences that training from scratch requires a lot of time of planning, and verification to make sure everything is working properly.

Documentation of your project
GitHub Location of Code
	
GIT HUB (does NOT include the models)
https://github.com/Jacob-C-Nguyen/ML483-ViVoice

MODELS
	F5_TTS_Base (1.3 GB)
https://huggingface.co/SWivid/F5-TTS

F5_TTS_Fine_Tuned (5.3 GB) (FOR FINE TUNED, ALSO INCLUDE CONFIG.JSON AND VOCAB.TXT from F5_TTS_Base)
https://drive.google.com/file/d/1Vy9UM1J4t2pmXsgg4uhQAUkmssk-_bHG/view?usp=sharing

Deployment and Setup Instructions
	
Download the github repository from the github link above, and either or both of the models for use in the program.

Download all of the required dependencies inside of the requirements.txt folder.

			Linux / WSL command:

			pip install -r requirements.txt
			pip install ft-tts
			pip install ffmpeg


Move the downloaded models into the “F5-TTS” folder located inside of the “Models” folder 

		Models
F5-TTS 
F5_TTS_Base
model.pt    config.json    vocab.txt
F5_TTS_Fine_Tuned
model.pt    config.json    vocab.txt

OPTIONAL
Create a 5-15 second voice clip (sampled to 24kHz) with a normalized transcription (recommended to put it into user/UserAudioAndTranscripts/Your_Folder_With_TXT_&_TRANS)
This is best done with open ai’s “whisper” tool, but it is not required.

Create a text file for the program to read out. (recommended location user/TextToTTS)


Steps to Run the Application

Navigate to the directory where you have downloaded the repo
Run the “Zero-Shot-NHP.py” with the command:
					python3 Zero-Shot-NHP.py
Follow the directions the CLI tool asks for


EX (Cloning Jeremy Clarkson’s Voice (Highlighted in yellow is user input)): 
--------------------------------------------------------------------------------------------------------------------


============================================================
F5-TTS Zero-Shot Voice Cloner
============================================================

Device Status:
  CUDA available: True
  GPU: NVIDIA GeForce RTX 4060 Laptop GPU

Available Models:
  1. F5TTS_Base
  2. F5TTS_Fine_Tuned

Select model (1-2): 2

 Using: F5TTS_Fine_Tuned

Reference audio file path: user/UserAudioAndTranscripts/jeremy/JC.wav
Reference audio transcript (or path to .txt file): user/UserAudioAndTranscripts/jeremy/JC.txt

Text to generate:
  Option 1: Type text directly
  Option 2: Load from .txt file
Choose (1 or 2): 2
Path to .txt file: user/TextToTTS/topgear.txt
Output filename (default: output.wav): jeremy.wav
Speed (default 1.0, range 0.5-2.0): 0.8
Temperature (default 0.3, range 0.1-1.0, lower = less hallucination): 0.3
Device (default cuda, options: cuda/cpu): cuda


--------------------------------------------------------------------------------------------------------------------


Wait for the F5-TTS to work its magic.
				
4.2. If you encounter troubles running the files later on, navigate to the “configs” folder in the f5–tts library and paste the .yaml file in the “model_helpers” folder from the github repo.

See your output in the “Outputs” folder












Resources
SWivid. (n.d.). F5-TTS/src/third_party at main · SWIVID/F5-TTS. GitHub. https://github.com/SWivid/F5-TTS/tree/main/src/third_party