Recognition of digits and characters
=========

```bash
digit-recognition.py --classifier-dir <dir> --img-dir <dir> [--teach] [--debug]
digit-recognition.py -c <dir> -i <dir> [-t] [-d]
```

During the teaching phase the program will highlight all the found digits/chars at a first stage. Then, by typing enter the learning phase will start and, one per time, the software will highlight the current digit/char and you have to teach to it what's the digit by typing the correspondent key on the keyboard.

During the testing phase the program will use the k-nearest-neighbor algorithm to match the current found digit with the real one.


##### Matteo Merola <mattmezza@gmail.com>
