import exifread
from PIL import Image
from spriteutils.spriteutil import SpriteSheet
from math import sqrt, degrees, acos, atan2
from numpy import array, zeros


class QRCode:
    """Represent a QR Code"""

    def __init__(self, image, position_detection_patterns):
        self._image = image
        self._position_detection_patterns = position_detection_patterns

    @property
    def image(self):
        """Image object of QR Code"""
        return self._image

    @property
    def position_detection_patterns(self):
        """Tuple of 3 outter sprites"""
        return self._position_detection_patterns


def get_vector(point_1, point_2):
    # Return the vector (point_1, point2) created by these 2 points
    return (point_2[0] - point_1[0], point_2[1] - point_1[1])


def get_magnitude(x, y):
    return round(sqrt(x**2 + y**2))


def find_qr_code_version(image, position_detection_patterns):
    """ Find version of the qr code basing on its number of modules

    @image: Image object
    @position_detection_patterns: tuples of 3 sprites objects that is also 3 edges
    of the QR code pattern

    return: version: int: version number of the qr code pattern
            theorical_number_of_modules: the corresponding number of module (theorical)
    """
    upper_left, upper_right, lower_left = position_detection_patterns
    upper_right_top_right = (upper_right.bottom_right[0], upper_right.top_left[1])
    sprite_width = upper_right.width
    x, y = get_vector(upper_right_top_right, upper_left.top_left)
    qr_witdh = get_magnitude(x, y)
    # One sprites contain 7 modules
    module_width = sprite_width / 7

    # number of modules found
    number_of_modules = qr_witdh / module_width
    version = round((number_of_modules - 21) / 4 + 1)
    # number of modules based on qr version
    theorical_number_of_modules = (version - 1) * 4 + 21
    return version, theorical_number_of_modules


def is_black_module(image_array, x, y, step):
    """ Analyze a module's pixels, if most pixels is black then the module
    is considered black

    Return True if the module is black, False otherwise
    """
    temp = []
    for i in range(x, x + step):
        for j in range(y, y + step):
            temp.append(image_array[i][j])
    return temp.count(0) >= temp.count(255)*1.0


def convert_qr_code_to_bit_array(image, position_detection_patterns):
    """ Convert the qr pattern to an array of 0 and 1, 1 for black module
    and 0 for white

    @image: Image object
    @position_detection_patterns: tuples of 3 sprites objects that is also 3 edges
    of the QR code pattern

    Return the bit array
    """
    upper_left, upper_right, lower_left = position_detection_patterns
    upper_right_top_right = (upper_right.bottom_right[0], upper_right.top_left[1])
    lower_left_bottom_left = (lower_left.top_left[0], lower_left.bottom_right[1])

    # Number of pixel of each sprite
    sprite_width = upper_right.width
    # Number of pixel of each module, each sprite contain 7 module
    module_width = round(sprite_width / 7)
    # Number of pixel of the whole qr pattern
    x, y = get_vector(upper_right_top_right, upper_left.top_left)
    qr_witdh = get_magnitude(x, y)
    number_of_modules = round(qr_witdh / module_width)

    # Turn Image object to numpy array object
    image_array = array(image)
    # Create an array (only 0) of dimension no_of_module x no_of_module
    bit_array = zeros((number_of_modules, number_of_modules), dtype=int)

    # Determine range of the qr pattern
    start_x =  upper_left.top_left[1]
    end_x =  lower_left_bottom_left[1]
    start_y = upper_left.top_left[0]
    end_y = upper_right_top_right[0]
    # Base on image array, draw the bit array
    count_row = 0
    for row in range(start_x, end_x, module_width):
        count_column = 0
        for column in range(start_y, end_y, module_width):
            if count_column == count_row == 2:
                is_black_module(image_array, row, column, module_width)
            try:
                if is_black_module(image_array, row, column, module_width):
                    bit_array[count_row][count_column] = 1
                count_column += 1
            except IndexError:
                break

        count_row += 1

    return bit_array


def find_qr_code(path_to_image, visible_threshold=50, square_threshold=0.3,
                 size_threshold=0.3, distance_threshold=0.3, position_threshold=0.3):
    """ Find the qr code pattern in image

    @path_to_image: str: image location in file system
    @thresholds: int and floats that corresponds to difference value threshold
    """
    try:
        image = monochromize_image(Image.open(path_to_image)).convert('L')
        sheet = SpriteSheet(image, background_color=255)
        sprites = list(sheet.find_sprites()[0].values())
        filtered_sprites = filter_square_sprites(filter_visible_sprites(sprites, visible_threshold), square_threshold)
        grouped_sprites = group_sprites_by_similar_size_and_distance(filtered_sprites, size_threshold, distance_threshold)
        position_detection_patterns = search_position_detection_patterns(grouped_sprites, position_threshold)
        outter_patterns = filter_matching_inner_outer_finder_patterns(position_detection_patterns)
        return [QRCode(image, pattern) for pattern in outter_patterns]

    except Exception:
        raise Exception("Please verify your argument and try again!")


def crop_qr_code_image(image, outter_sprites):
    """ The function returns a PIL.Image corresponding to the
    portion of the QR code rotated and cropped from image

    @image: PIL.Image object
    @outter_sprites: tuple of 3 Sprite objects, in exact order (upper_left_sprite,
    upper_right_sprite, lower_left_sprite)

    @return: Image object
    """

    upper_left_sprite, upper_right_sprite, lower_left_sprite = outter_sprites

    upper_left_center = upper_left_sprite.centroid
    upper_right_center = upper_right_sprite.centroid
    lower_left_center = lower_left_sprite.centroid
    qr_pattern_center = ((upper_right_center[0] + lower_left_center[0]) // 2, (upper_right_center[1] + lower_left_center[1]) // 2)
    x, y = get_vector(upper_left_center, upper_right_center)
    # Distance between two center point of upper_left and upper_right
    edge_distance = get_magnitude(x, y)
    trasition_from_center = edge_distance // 2

    # Calcul new center coordinates
    new_upper_left_center = (qr_pattern_center[0] - trasition_from_center, qr_pattern_center[1] - trasition_from_center)
    new_lower_right_center = (qr_pattern_center[0] + trasition_from_center, qr_pattern_center[1] + trasition_from_center)

    # Transition from center to corner
    trasition_value = upper_left_sprite.width // 2
    new_upper_left_top_left = (new_upper_left_center[0] - trasition_value, new_upper_left_center[1] - trasition_value)
    new_lower_right_bottom_right = (new_lower_right_center[0] + trasition_value, new_lower_right_center[1] + trasition_value)


    # Vector from center of qr pattern to original upper left center
    vector_center_to_original = get_vector(qr_pattern_center, upper_left_center)
    # Vector from center of qr pattern to new upper left center
    vector_center_to_new = get_vector(qr_pattern_center, new_upper_left_center)

    # Math formula
    dot_product = vector_center_to_original[0]*vector_center_to_new[0] + vector_center_to_original[1]*vector_center_to_new[1]
    determinant = vector_center_to_original[0]*vector_center_to_new[1] - vector_center_to_original[1]*vector_center_to_new[0]

    angle = 360 - degrees(atan2(determinant, dot_product))
    rotated_image = image.rotate(angle, fillcolor=255, center=qr_pattern_center)

    # Crop area
    left = new_upper_left_top_left[0]
    upper = new_upper_left_top_left[1]
    right = new_lower_right_bottom_right[0]
    lower = new_lower_right_bottom_right[1]
    cropped_image = rotated_image.crop((left, upper, right, lower))

    return cropped_image


def filter_square_sprites(sprites, similarity_threshold):
    """ Remove all sprites that are not considered as square

    @sprites: A list of Sprite objects.
    @similarity_threshold: A float number between 0.0 and 1.0 of the relative
    difference of the width and height of the sprite's boundary box over which
    the sprite is not considered as a square.

    @return: A list of square-filtered Sprite objects
    """
    try:
        # Catch invalid similarity threshold value
        if 0.0 >= similarity_threshold >= 1.0 or not isinstance(similarity_threshold, float):
            raise("Argument similarity_threshold MUST be a float number of which value fall between 0.0 and 1.0 ! Please try again")
            return
        # Compare sprite's height/width ratio and the threshold
        compare_function = lambda sprite : sprite if (abs(sprite.width - sprite.height) / sprite.height <= similarity_threshold or abs(sprite.height - sprite.width) / sprite.width <= similarity_threshold) else 0
        reformatted_sprites = set(map(compare_function, sprites))
        # Remove invalid sprites from set
        if 0 in reformatted_sprites:
            reformatted_sprites.remove(0)
        return list(reformatted_sprites)
    except Exception as e:
        raise Exception("Error! Make sure that your arguments are valid")


def filter_matching_inner_outer_finder_patterns(finder_patterns):
    """ Match inner and outer pattern from a list of patterns

    @finder_patterns: list of tuple. Each tuple corresponds to 3 sprites
    that may possibly correspond to a finder pattern of a QR code

    @Return: a list of tuples corresponding to the outer finder patterns.
    """

    def is_outer_inner(current_pattern, next_pattern):
        """ Check if next_pattern is inside current_pattern

        @current_pattern: next_pattern: tuple of Sprite objects
        @Return: Boolen, True if current_pattern is the outer
        """

        # Check each correspond pair of Sprites
        for index in range(0, 3):
            current_sprite = current_pattern[index]
            next_sprite = next_pattern[index]
            # Check if next_pattern's points are totally nested inside current_sprite's border
            if not (current_sprite.top_left[0] < next_sprite.top_left[0] < next_sprite.bottom_right[0] < current_pattern[index].bottom_right[0] and
                    current_sprite.top_left[1] < next_sprite.top_left[1] < next_sprite.bottom_right[1] < current_pattern[index].bottom_right[1]):
                return False
        return True

    outer_finder_patterns = []

    for index, pattern in enumerate(finder_patterns):
        for next_index in range(index + 1, len(finder_patterns)):
            current_pattern = finder_patterns[index]
            next_pattern = finder_patterns[next_index]
            # Outer pattern is not yet defined
            outer_pattern = None

            # Check if they are inner_outer pair, and which one is outer
            if is_outer_inner(current_pattern, next_pattern):
                outer_pattern = current_pattern
            elif is_outer_inner(next_pattern, current_pattern):
                outer_pattern = next_pattern

            if outer_pattern:
                outer_finder_patterns.append(outer_pattern)

    return outer_finder_patterns


def search_position_detection_patterns(sprite_pairs, orthogonality_threshold):
    """ Search for tuples of 3 sprites that may possibly correspond to a finder pattern of a QR code,
    meaning they form an angle almost orthogonal.
    @arg: sprite_pairs: A list of pairs of sprites.
    @arg: orthogonality_threshold: A float number between 0.0 and 1.0
    of the relative difference between the angle of two pairs of sprites
    less or equal which the two pairs are considered orthogonal.

    @return: list of tuples, each tuples contain 3 Sprite objects
    """
    def get_common_sprite(this_pair, that_pair):
        """
        Find the common sprite from the pair
        @this_pair: that_pair: Tuple that contains 2 'same-size-and-same-distance' sprites
        @return: The common sprite if exist. False otherwise.
        """
        common_sprite = set(this_pair) & set(that_pair)
        for sprite in common_sprite:
            return sprite
        return False

    def get_other_sprite(common_sprite, pair):
        # Return the sprite other than the common one
        return [sprite for sprite in pair if sprite is not common_sprite][0]

    def get_angle_between_two_sprite_pairs(common_sprite, sprite_1, sprite_2):
        """ Return angle and angle sign of 2
        """
        # Always use center points as a preference of a sprite
        common_point_center = common_sprite.centroid
        center_1 = sprite_1.centroid
        center_2 = sprite_2.centroid

        #Vectorial math formula
        vector_1_x, vector_1_y = get_vector(common_point_center, center_1)
        vector_2_x, vector_2_y = get_vector(common_point_center, center_2)

        dot_products = vector_1_x*vector_2_x + vector_1_y*vector_2_y

        magnitude_vector_1 = get_magnitude(vector_1_x, vector_1_y)
        magnitude_vector_2 = get_magnitude(vector_2_x, vector_2_y)
        # Cosinus math formula
        cosinus_value = round(dot_products / (magnitude_vector_1 * magnitude_vector_2), 2)

        z_component = vector_1_x * vector_2_y - vector_1_y * vector_2_x
        angle_sign = round(z_component / abs(z_component))

        angle = degrees(acos(cosinus_value))

        return round(angle), angle_sign

    possible_patterns = []
    for pairs in sprite_pairs:
        for index, pair in enumerate(pairs):
            number_of_pairs = len(pairs)
            for next_index in range(index + 1, number_of_pairs):
                next_pair = pairs[next_index]
                common_sprite = get_common_sprite(pair, next_pair)
                # Ignore pairs that dont have a common sprite,
                # because 4 unconnected sprite cannot form a triangle
                if not common_sprite:
                    continue
                sprite_1 = get_other_sprite(common_sprite, pair)
                sprite_2 = get_other_sprite(common_sprite, next_pair)
                try:
                    angle, angle_sign = get_angle_between_two_sprite_pairs(common_sprite, sprite_1, sprite_2)
                except Exception as e:
                    continue
                # Ignore those 3 sprites that don't form a right triangle (90degree)
                if not abs(angle - 90) / 90 <= orthogonality_threshold:
                    continue
                # Reorder the three sprite in order: top-left, top-right, bottom-left
                if angle_sign < 0:
                    common_sprite, sprite_1, sprite_2 = common_sprite, sprite_2, sprite_1
                # Add tuple to the result list
                possible_patterns.append((common_sprite, sprite_1, sprite_2))
    return possible_patterns


def group_sprites_by_similar_size_and_distance(sprites, similar_size_threshold, similar_distance_threshold):
    """ Group sprites by its size and distance
    @sprites: A list of Sprite objects.
    @similar_size_threshold: A float number between 0.0 and 1.0 representing the
    relative difference between the sizes (surface areas)
    of two sprites below which these sprites are considered similar.
    @similar_distance_threshold: A float number between 0.0 and 1.0
    representing the relative difference between the distances from
    the sprites of 2 pairs below which these pairs are considered having similar distance.
    @Return a list of groups (lists) of pairs of sprites with the following properties:
        1.the sprites of a same group have similar size.
        2.the distance between the sprites of each pair of a same group is equivalent.
    """
    def add_to_group(group, element_to_add, comparing_element, threshold):
        already_exist = False
        for element in group:
            if element == 0:
                continue
            if (abs(comparing_element - element) / element <= threshold):
                group[element].append(element_to_add)
                already_exist = True
        if not already_exist:
            group[comparing_element] = [element_to_add]
    try:
        # Catch invalid arguments
        if (0.0 >= similar_size_threshold >= 1.0
            or not isinstance(similar_size_threshold, float)
            or 0.0 >= similar_distance_threshold >= 1.0
            or not isinstance(similar_distance_threshold, float)):
            raise Exception("Argument similar_size_threshold and similar_distance_threshold MUST be a float number of which value fall between 0.0 and 1.0 ! Please try again")
            return
        # Group sprites by their sizes
        size_groups = {}
        for sprite in sprites:
            add_to_group(size_groups, sprite, sprite.surface, similar_size_threshold)

        result_group = []
        for size in size_groups:
            distance_groups = {}
            # Skip groups that have less than 3 elements
            if len(size_groups[size]) < 3:
                continue
            for index, sprite in enumerate(size_groups[size]):
                # Find center pixels
                origin_sprite_center = sprite.centroid
                for next_element in range(index + 1, len(size_groups[size])):
                    destination_sprite = size_groups[size][next_element]
                    destination_sprite_center = destination_sprite.centroid
                    # Calcul distance
                    difference_in_x = origin_sprite_center[0] - destination_sprite_center[0]
                    difference_in_y = origin_sprite_center[1] - destination_sprite_center[1]
                    sprites_distance = round(sqrt(difference_in_x * difference_in_x + difference_in_y * difference_in_y))
                    add_to_group(distance_groups, (sprite, destination_sprite), sprites_distance, similar_distance_threshold)
            for key in distance_groups:
                if len(distance_groups[key]) >= 2:
                    result_group.append(distance_groups[key])
        return result_group

    except Exception:
        raise Exception("Error! Make sure that your arguments are valid")


def filter_dense_sprites(sprites, density_threshold):
    """ Remove all sprites whose density is not dense enough

    @sprites: list of Sprite objects
    @density_threshold: A float number between 0.0 and 1.0 representing
    the relative difference between the number of pixels of a sprite and
    the surface area of the boundary box of this sprite,
    over which the sprite is considered as dense.

    @return: filtered list of Sprite's objects
    """
    try:
        # Catch invalid density threshold value
        if 0.0 > density_threshold > 1.0 or not isinstance(density_threshold, float):
            raise Exception("Argument similarity_threshold MUST be a float number of which value fall between 0.0 and 1.0 ! Please try again")
            return
        # Compare sprite's height/width ratio and the threshold
        compare_function = lambda sprite : sprite if sprite.density >= density_threshold else 0
        reformatted_sprites = set(map(compare_function, sprites))
        # Remove invalid sprites from set
        if 0 in reformatted_sprites:
            reformatted_sprites.remove(0)
        return list(reformatted_sprites)
    except Exception:
        raise Exception("Error! Make sure that your arguments are valid")


def filter_visible_sprites(sprites, min_surface_area):
    """ Remove all sprite objects have smaller surface than minimum requirement

    @sprites: list of Sprite objects
    @min_surface_area: An integer of minimum surface requirement

    Return filtered list of sprites objects
    """
    try:
        if not isinstance(min_surface_area, int):
            raise Exception("Argument min_surface_area should be an integer! Please try again")
            return
        # Compare sprite's surface to minimum requirement, invalid sprites are marked as 0
        compare_function = lambda sprite : sprite if sprite.surface >= min_surface_area else 0
        reformatted_sprites = set(map(compare_function, sprites))
        # Remove invalid sprites from set
        if 0 in reformatted_sprites:
            reformatted_sprites.remove(0)
        return list(reformatted_sprites)
    except Exception as e:
        raise Exception("Error! Make sure that your arguments are valid")


def calculate_brightness(image):
    """Get the brightness value of an image

    @image: an Image object

    @return: float number between 0.0 et 1.0 of the brightness value
    """
    try:
        greyscale_image = image.convert('L')
        # Histogram - List of number of pixels corresponding to it's brightness value
        histogram = greyscale_image.histogram()
        # Total pixels number
        pixels = sum(histogram)
        brightness = scale = len(histogram)

        for index in range(0, scale):
            # ratio between the number of pixels having this color value and the total number of pixels
            ratio = histogram[index] / pixels
            # recalculate brightness each time we analyze the next scale
            delta = ratio * (index - scale)
            brightness += delta

        return 1.0 if brightness == 255 else brightness / scale

    except Exception:
        raise Exception("Error! Make sure that your arguments are valid")


def monochromize_image(image, brightness=0.5):
    """ Turn an image to a monochromized version

    @image: An Image object
    @brightness: float between 0.0 and 1.0

    @return monochromized_image: Image object of the monochromized version
    """

    try:
        # Catch invalid brightness value
        if 0.0 >= brightness >= 1.0 or not isinstance(brightness, float):
            raise Exception("Your brightness argument is not valid! It should be a float between 0.0 and 1.0")
            return

        # Threshold equation
        threshold = 255 * brightness
        # Working with Black and White version, compare between pixel's value and threshold
        # in order to change pixel to correct monochrome value.
        compare_to_threshold = lambda x : 255 if x > threshold else 0
        monochromized_image = image.convert('L').point(compare_to_threshold, mode='1')

        return monochromized_image

    except Exception:
        raise Exception("Error! Make sure that your arguments are valid")


def load_image_and_correct_orientation(file_path_name):
    """ Read image's Exif data then rorate the image if needed

    @file_path_name: file location

    @return image: A (rotated) Image object
    """
    try:
        image = Image.open(file_path_name)
        tags = {}

        with open(file_path_name, 'rb') as opened_file:
            # Read Exif tags until find 'Image Orientation' keyword
            tags = exifread.process_file(opened_file, stop_tag='Image Orientation')

        if "Image Orientation" in tags.keys():
            orientation = tags["Image Orientation"]
            value = orientation.values
            # Rotate image depend on current orientation value
            if 5 in value:
                value += [2, 6]
            if 7 in value:
                value += [2, 8]

            if 6 in value:
                image = image.transpose(Image.ROTATE_270)
            if 8 in value:
                image = image.transpose(Image.ROTATE_90)
            if 3 in value:
                image = image.transpose(Image.ROTATE_180)

            if 2 in value:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if 4 in value:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)

        return image

    except Exception:
        raise Exception("Error! Make sure that your arguments are valid")
