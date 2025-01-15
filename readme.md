# auto_mpg ML tutorial

## Windows

### install python

- Currently, Python 3.12 is the latest version in which tensorflow is available (on 2025,Jan,15th)
- install python from Official installer
- add Python Path to Environment.
- on git bash

```bash
python -V # confirm python
cd path/to/auto_mpg
python -m venv venv
source ./venv/Script/activate
pip install tensorflow
pip install pandas
pip install scikit-learn
```

### execute

```bash
python build.py       # make trained model & scaler setting 
python estimate.py    # execute estimation using model above
python estimate_tk.py # most simple gui
```


