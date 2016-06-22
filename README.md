Recognition of digits and characters
=========

```python
python teach.py img/source/folder classifier/onlydigits

python test.py img/test/folder classifier/onlydigits
```

To teach also letters please run the command with the `--with-letters` parameter

```python
python teach.py img/source/folder classifier/digitsandletters --with-letters

python test.py img/test/folder classifier/digitsandletters
```

During the teaching phase the program will highlight all the found digits/chars at a first stage. Then, by typing enter the learning phase will start and, one per time, the software will highlight the current digit/char and you have to teach to it what's the digit/char by typing the correspondent key on the keyboard.

During the testing phase the program will use the k-nearest-neighbor algorithm to match the current found digit/char with the real one.


##### Matteo Merola <mattmezza@gmail.com>
