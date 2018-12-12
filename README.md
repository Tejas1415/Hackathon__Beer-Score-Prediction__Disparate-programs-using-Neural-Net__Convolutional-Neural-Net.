## Hackathon-Beer-Score-Prediction-programs-using-Neural-Net-Convolutional-Neural-Net

/*
Coded by Tejas Krishna Reddy 
Given certain number of features including both categorical and numerical data, a score had to be predicted.
/*

The number of different categorical features in a feature set were observed to be huge. Therefore, if one-hot encoding technique was used to convert categorical variables, the number of features would have exceeded 10k which would make it an high dimensional dataset increasing the computational complexity to build the model.

Therefore, "Binary Encoding Technique" was practically implemented in the program (Built from scratch). This resonably reduced the number of features.

Rather than using typical GBR, XGBoost or RandomForest techniques for Regression in hackathons, Neural Net and CNN was used to model it. Experimentally it was found that GBR and Extratreesregressor gave approximately similar results as CNN which was around 40% accuracy where the highest in the hackathon was 48% accuracy. 

All kinds of regression techniques were also experimented over this data for the sake of knowledge, and they are uploaded in seperate files.

For any kind of doubts, feel free to contact me at tejastk.reddy@gmail.com
