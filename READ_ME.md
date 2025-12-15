
Use the following command on linux to install all python libraries:
    pip install -r requirements.txt




Please insert your models in folders into the folder inside of the "Models" folder (F5-TTS).

Your file structure should look like this:


    Models
        F5-TTS 
            F5_TTS_Base
                model.pt    config.json    vocab.txt
            F5_TTS_Fine_Tuned
                model.pt    config.json    vocab.txt



You can download the pretrained model from 
    "https://huggingface.co/SWivid/F5-TTS"


    OR


you can download the my fine tuned model from model from 
    "https://drive.google.com/file/d/1Vy9UM1J4t2pmXsgg4uhQAUkmssk-_bHG/view?usp=sharing" 

    (both use the same config.json and vocab.txt)
