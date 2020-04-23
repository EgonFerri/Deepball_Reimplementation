# Deepball_Reimplementation

# Neural Networks for Data Science Applications (2019/2020)
## Final exam 

* **Student**: Egon Ferri (1700963).
* **Reference paper / topic**: Jacek Komorowski, Grzegorz Kurzejamski, Grzegorz Sarwas, 2019.

    DeepBall: Deep Neural-Network Ball Detector and FootAndBall: Integrated player and ball detector.
 
The paper describes the implementation of a neural network with the scope to dectet ball in long shot HD videos of football matches.
The network is fully convolutional, and produces confindence maps that encode the position of the detected ball.
A part from the confidence map, the main "new" technical feature is represented by the use of the hypercolumn concept;
feature maps at different deep level are upsampled to match dimensions, joined and fed all together to the last convolutional layer.

The idea is to keep the network simple and a relatively low number of parameters, in order to be able to process video very fast.
 
<p align="center">
  <img src="gif_mini.gif">
</p>

