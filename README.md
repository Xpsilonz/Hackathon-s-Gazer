# Hackathon-s-Gazer

Welcome to our repo for our perplexity's sonar hackathon submission, which is an  Eye-Tracking and Text Recognition System. Then you can use this stored text data to start a chat using the Perplexity Sonar API. 

## Reccomended Python Version

#### 3.10 (To download click this link) -> [Download Python 3.10.9](https://www.python.org/downloads/release/python-3109/)


## Installations 


### 1. Clone the Repository

Open your terminal and run the following command to clone the repository

   ```
   git clone https://github.com/Xpsilonz/Hackathon-s-Gazer.git
   ```

Then Navigate to the Project Directory

```
cd Hackathon-s-Gazer
```

### 2. Install Tesseract OCR

You need to install Tesseract OCR on your system. Follow the instructions based on your operating system: 

_For Windows_

1. [Download Tesseract OCR](https://github.com/tesseract-ocr/tesseract/wiki)
2. Run the installer and follow the setup instructions.
3. Add the Tesseract installation path (e.g., C:\Program Files\Tesseract-OCR) to your system's PATH environment variable.

_For macOS_

You can install Tesseract using Homebrew

```
brew install tesseract
```

_For Linux_

You can install Tesseract using your package manager. For example, on Ubuntu

```
sudo apt-get install tesseract-ocr
```

### 3. Install Python Dependencies

Make sure you have Python installed. You can install the necessary packages using PIP

```
pip install -r requirements.txt
```


### Or


### 1.Download a zip file

On the repository page, look for the green "Code" button located near the top right corner.
Click on the "Code" button, and a dropdown menu will appear.
In the dropdown menu, you will see an option that says "Download ZIP." Click on it.

### 2.Install Python Dependencies

Make sure you have Python installed. You can install the necessary packages using PIP

```
pip install -r requirements.txt
```

## How to Use

### 1.Run the Tracker: After installing the dependencies, you can start the eye-tracking application by running

```
python tracker.py
```

replace python with py -3.10 (if you have multiple versions)

### 2. Recommend reading an instruction before using by pressing ESC + i

| Key   | Action                           |
| ----- | -------------------------------- |
| `c`   | Manually train calibration model |
| `o`   | Toggle OCR on/off                |
| `i`   | Show instructions               |
| `x`   | Show recent recognized texts     |
| `f`   | Toggle fullscreen                |
| `h`   | Show/hide OCR region frame       |
| `ESC` | Open/close menu                  |
| `q`   | Quit                             |


## Data Utilization

The text you look at from the tracker program will be sent to a remote database. This data can be utilized to enhance the functionality of a chatbot on the extension. By analyzing the text you frequently engage with, the chatbot can provide more relevant and personalized information based on your interests and needs.

**Architecture**

[ Eye Tracker ]

      ↓
      
[ OCR Recognition ]

      ↓
      
[ JSON log + HTTP POST ]

      ↓
      
[ Extension/Server with Perplexity API ]


## Setting Up Chrome Extension

To intergrate the tracking system with Sonar API chatbot you can download the Chrome extension that works with this system. Follow these steps to install the extension: 

### 1.Download the Extension 

Download the extension folder from the repository. You can find it in the extension directory.

### 2.Open Chrome and Access Extensions

Open Google Chrome.
Go to the Extensions page by entering chrome://extensions/ in the address bar.  

### 3.Enable Developer Mode

Toggle the "Developer mode" switch in the top right corner of the Extensions page.

### 4.Load the Extension

Click on the "Load unpacked" button.
Select the folder where you downloaded the extension. Once loaded, the extension will be available for use alongside the Eye-Tracking and Text Recognition System.

## Contributing

We welcome contributions to improve the Eye-Tracking and Text Recognition System! If you have suggestions or enhancements, please feel free to submit a pull request or open an issue.

## LICENSE

This project is licensed under the MIT License. See the ![LICENSE](LICENSE) file for more details.  
