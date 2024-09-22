This code is the computer vision system developed for my university's Very Small Size Robot Soccer team.

I have used Python primarily with the OpenCV library, and presented the results as part of my graduation essay in 2019.

One of the challenges in robot soccer is identifying the elements on the field, which is usually achieved by means of colour markings on the robots. However, when subjected to different lighting conditions or the shadows cast by the audience, the system must understand that a darker or brighter shade of a colour still counts as the same. To achieve such a goal, my system would identify the colours selected by the user on screen from the camera output, and calculate a range of the visible spectrum of that colour within which it could consider it to be the same. This is achieved by extrapolating a certain range around the colour value, implemented as HSV or RGB codifications.

The purpose of the system was to provide these results in a fast enough speed so that the path planning algorithm could take quick decisions to play the game. Furthermore, my graduation essay used this same system adapted to the RGB and HSV codifications, in order to decide which one was more accurate. The HSV system was deemed more precise by a significant margin and implemented on the final robot team.
