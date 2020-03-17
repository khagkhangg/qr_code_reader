## Introduction:
&nbsp; The project is generally used for detecting and analyzing a QR codes. Further information abouts its function can be found in ./mission folder.


## Installation:
The project require Python 3.7+ to run

##### &emsp; Step 1: Clone or Downloads the project, using this command:

     git clone https://github.com/intek-training-jsc/qr-code-reader-khangvu12-1
##### &emsp; Step 2: Go to project folder:

     Go to project folder using:
          cd [path_to_project]/qr-code-reader-khangvu12-1

##### &emsp; Step 2: Install required libs and tools, using this command in Terminal:

     pip3 install -r requirement.txt

## Example of usage:

     from detector import *

     image = monochromize_image(Image.open('./mini.png')).convert('L')
     sheet = SpriteSheet(image, background_color=255)
     sprites = list(sheet.find_sprites()[0].values())
     a = filter_square_sprites(filter_visible_sprites(sprites, 500), 0.2)
     b = group_sprites_by_similar_size_and_distance(a, 0.2, 0.05)
     c = search_position_detection_patterns(b, 0.4)
     d = filter_matching_inner_outer_finder_patterns(c)
     e = crop_qr_code_image(image, d[0])
     e.show()

     list_qrs = find_qr_code(path_to_image)
     version, width = find_qr_code_version(list_qrs[0].image, list_qrs[0].position_detection_patterns)
     print(version, width)

     bit_array = convert_qr_code_to_bit_array(list_qrs[0].image, list_qrs[0].position_detection_patterns)
     print(bit_array)

## Contact:
&emsp;&emsp;&emsp; During the usage of the project, if you have any question, please contact me personally at INTEK HCM City.

## Contributors:
&emsp;&emsp;&emsp; Khang VU from INTEK Institute, HCM City
