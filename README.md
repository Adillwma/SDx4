
<div align="center">

# Windows SDx4 Image Upscaler
### Author: Adill Al-Ashgar
#### Stability AI's Stable Diffusion x4 Upscaler in Windows GUI Package with advanced features.

<img src="Images/SDX4_BANNER.png" width="800"> 

    - 4x neural engine image upscaling
    - No internet connection required
    - Local processing, none of your data is sent to any third party.

[![Github Repo](https://img.shields.io/badge/GitHub_Repo-SDx4_ImageUpscaler-yellow.svg)](https://github.com/Adillwma/Windows_SDx4_ImageUpscaler)
[![Language](https://img.shields.io/badge/language-Python-blue.svg)](https://www.python.org/)
[![Published](https://img.shields.io/badge/Published-2023-purple.svg)]()
</div>

## Introduction
SDx4 Image Upscaler is a windows GUI package for the [Stability AI Stable Diffusion x4 Upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler) model. It is designed to be a simple to use, lightweight, and secure way to upscale images using the power of AI. It is packaged as a windows installer with no python or code dependencies. It is designed to be a simple to use, lightweight, and secure way to upscale images using the power of AI. It is packaged as a windows installer with no python or code dependencies. It is designed to be a simple to use, lightweight, and secure way to upscale images using the power of AI. It is packaged as a windows installer with no python or code dependencies. It is designed to be a simple to use, lightweight, and secure way to upscale images using the power of AI. It is packaged as a windows installer with no python or code dependencies. It is designed to be a simple to use, lightweight, and secure way to upscale images using the power of AI. It is packaged as a windows installer with no python or code dependencies. It is designed to be a simple to use, lightweight, and secure way to upscale images using the power of AI. It is packaged as a windows installer with no python or code dependencies. It is designed to be a simple to use, lightweight, and secure way to upscale images using the power of AI. It is packaged as a windows installer with no python or code dependencies.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contributions](#contributions)
- [Contact](#contact)

# Features
- Packaged as a windows installer with no python or code dependencies.
- Lightweight modern windows GUI to simplify usage. 
- Batch processing upscaling ability
- Icorperates tiled processing to allow for large images to be upscaled with low ram usage and direct cpu processig for users without cuda gpu's.
- Dynamic tileshifting to reduce edge artifacts, and allow for a more accurate upscale whilst avoiding dark pixel padding or processing non image data.
- Soft and hard edge blending selectable by user.
- Ability to use the xFormers library to greatly speed up processing given a cuda gpu.
- Ability to use cpu offloading to allow for boosted cpu processing whilst using a cuda gpu.
- Ability to introduce attention slicing to reduce memory usage (not recomended, inferior to xFormers and covered by tiled processing) 
- Local processing, no internet connection required.
- Data security, none of your data is sent to any third party.

## Installation
BackupInspector is primarily developed for Windows users, and we provide a pre-packaged executable for easy installation on Windows. However, it can also be run on other operating systems by executing the Python code directly.

### Windows
To install Windows_SDx4, follow these steps:

1. Download the latest release of Windows_SDx4.exe from this repo using the following link: [BackupInspector Download](https://github.com/Adillwma/Windows_SDx4/raw/main/Windows_SDx4.exe)

2. Run the downloaded BackupInspector.exe file.

### Other Operating Systems













Built on top of the [Stability AI Stable Diffusion x4 Upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler) model:

Developed by: Robin Rombach, Patrick Esser 
https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler
License: CreativeML Open RAIL++-M License
Resources for more information: GitHub Repository.

@InProceedings{Rombach_2022_CVPR,
    author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn},
    title     = {High-Resolution Image Synthesis With Latent Diffusion Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {10684-10695}
}





PyQt6 is released under the GPL v3


















my liscence
[Your Software Name] License
Version 1.0, [Date]

Copyright (c) [Year], [Your Company/Organization Name]

CreativeML Open RAIL++-M License
dated November 24, 2022

The [Your Software Name] software and its accompanying documentation are licensed under the terms and conditions of the CreativeML Open RAIL++-M License. By using, reproducing, or distributing the software, you agree to be bound by the terms of this license.

1. Original License:
   The [Your Software Name] software is based on the CreativeML Open RAIL++-M License dated November 24, 2022. The full text of the original license is available in the LICENSE file accompanying this software or can be found at [link to the original license].

2. Modifications:
   Any modifications or additions made to the original software, including the creation of a graphical user interface (GUI) bundled as an executable, are subject to the terms of the CreativeML Open RAIL++-M License.

3. Use-Based Restrictions:
   The software includes use-based restrictions as specified in Attachment A of the CreativeML Open RAIL++-M License. Users are prohibited from using the software in ways that violate applicable laws or the specified restrictions.

4. Notices:
   Users must be made aware that the software includes components covered by the CreativeML Open RAIL++-M License. Notices must be provided in the software interface or accompanying documentation.

5. Disclaimer and Limitation of Liability:
   The [Your Software Name] software is provided "AS IS" without any warranties or conditions. The distributor, [Your Company/Organization Name], is not liable for any damages arising from the use of the software.

6. Compliance with Applicable Laws:
   Users are required to comply with all applicable national, federal, state, local, or international laws and regulations when using the software.

7. Distribution:
   If you distribute the [Your Software Name] software, you must provide access to the full text of the CreativeML Open RAIL++-M License and any additional documentation.

This [Your Software Name] License is in effect as of [Date]. For any questions regarding this license, please contact [Your Contact Information].

END OF LICENSE




















