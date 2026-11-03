Step 1: Sentiment labels (lexicon-based) [ only for yooutube]

Use VADER. It needs no training, and it handles YouTube-style text (caps, “!!!”, emojis, slang) well.


Theme classifier

GIven glassdoor dastet is fixed we will comapre performane between 2 moels 

1. BERT - allows the model to dicover potential theme that are not tagged out 
2. Zero-Shot Approach - using the current predefiend themes in them-congig.py to explore the result 


now i have changed and uodate the clean.py where i added more cleaning layer and method and  , we will continue on with Theme label: for glassdoor datatset ,

GIven i had changes the structure in clean.oy , makre sure all the other folder ca still ru clean.py , the final call of this willl be in / notbook/glassdoor.ipynb. For now ,I want to compre  the performance of The Zero-Shot Approach and BERT see which one is better  interms of text engine , if BERT is giving simialr reuslt then Zero-Shot Appoach will be the best idea . Also keep in mind that i have i have no limit for theme label , emaning a text can have multple label without any max limit 


--- plot result into --- 

Before we start with theme label. I want you to note down the current performace of them categorization ,good kickstart will be looking through the /notebook/glassdoor.ipynb , obeserve the data pattern in there , the result , include theme distributon across all four column, and top 20 most common words across all column .  note it down in a new doc under  in /notebook/theme-categorization-result.md , put this foler in gitignore


--TMR 8 OCT to do -

[] work on the model comparison between BERT and Zero_shot-Approach
[] observe the result and see if any changes is needed
[] apply 
[]