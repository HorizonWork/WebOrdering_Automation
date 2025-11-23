def replace_special_characters(text, mode='replace'):
    """
    Xóa hoặc thay thế các ký tự đặc biệt trong văn bản
    
    Args:
        text (str): Văn bản đầu vào
        mode (str): Chế độ xử lý - 'replace' để thay thế, 'remove' để xóa
    
    Returns:
        str: Văn bản đã xử lý
    """
    # Từ điển ánh xạ các ký tự đặc biệt thành từ ngữ thông thường
    char_mapping = {
        '✅': 'yes',
        '❌': 'no',
        '✔': 'yes',
        '✗': 'no',
        '✓': 'yes',
        '✘': 'no',
        '☀': 'sunny',
        '☁': 'cloudy',
        '☂': 'rainy',
        '☃': 'snowy',
        '⚡': 'lightning',
        '❤': 'love',
        '🔥': 'fire',
        '🌟': 'star',
        '💯': 'perfect',
        '🎉': 'celebration',
        '👏': 'clap',
        '🙌': 'raise_hands',
        '👍': 'thumbs_up',
        '👎': 'thumbs_down',
        '👌': 'ok',
        '🙏': 'pray',
        '👀': 'eyes',
        '🐶': 'dog',
        '🐱': 'cat',
        '🐭': 'mouse',
        '🐹': 'hamster',
        '🐰': 'rabbit',
        '🦊': 'fox',
        '🐻': 'bear',
        '🐼': 'panda',
        '🐨': 'koala',
        '🦁': 'lion',
        '🐯': 'tiger',
        '🐮': 'cow',
        '🐷': 'pig',
        '🐸': 'frog',
        '🐵': 'monkey',
        '🐔': 'chicken',
        '🐧': 'penguin',
        '🐦': 'bird',
        '🦆': 'duck',
        '🦅': 'eagle',
        '🦉': 'owl',
        '🦇': 'bat',
        '🐺': 'wolf',
        '🐗': 'boar',
        '🐴': 'horse',
        '🦄': 'unicorn',
        '🐝': 'bee',
        '🐛': 'bug',
        '🦋': 'butterfly',
        '🐌': 'snail',
        '🐞': 'ladybug',
        '🐜': 'ant',
        '🦟': 'mosquito',
        '🦗': 'cricket',
        '🕷': 'spider',
        '🦂': 'scorpion',
        '🐢': 'turtle',
        '🐍': 'snake',
        '🦎': 'lizard',
        '🦖': 't-rex',
        '🦕': 'sauropod',
        '🐙': 'octopus',
        '🦑': 'squid',
        '🦐': 'shrimp',
        '🦞': 'lobster',
        '🦀': 'crab',
        '🐡': 'blowfish',
        '🐠': 'tropical_fish',
        '🐟': 'fish',
        '🐬': 'dolphin',
        '🐳': 'whale',
        '🐋': 'whale2',
        '🦈': 'shark',
        '🐊': 'crocodile',
        '🐅': 'tiger2',
        '🐆': 'leopard',
        '🦓': 'zebra',
        '🦍': 'gorilla',
        '🐘': 'elephant',
        '🦏': 'rhino',
        '🦛': 'hippo',
        '🐪': 'dromedary_camel',
        '🐫': 'camel',
        '🦒': 'giraffe',
        '🦘': 'kangaroo',
        '🐃': 'water_buffalo',
        '🐂': 'ox',
        '🐄': 'cow2',
        '🐎': 'racehorse',
        '🐖': 'pig2',
        '🐏': 'ram',
        '🐑': 'sheep',
        '🦙': 'llama',
        ' goats': 'goat',  # Lỗi đánh máy cần sửa: ' goats' -> 'goat'
        '🦌': 'deer',
        '🐕': 'dog2',
        '🐩': 'poodle',
        '🦮': 'guide_dog',
        '🐕‍🦺': 'service_dog',
        '🐈': 'cat2',
        '🐓': 'rooster',
        '🦃': 'turkey',
        '🦚': 'peacock',
        '🦜': 'parrot',
        '🦢': 'swan',
        '🦩': 'flamingo',
        '🕊': 'dove',
        '🐇': 'rabbit2',
        '🦝': 'raccoon',
        '🦨': 'skunk',
        '🦡': 'badger',
        '🦦': 'otter',
        '🦥': 'sloth',
        '🐁': 'mouse2',
        '🐀': 'rat',
        '🐿': 'chipmunk',
        '🦔': 'hedgehog',
    }
    
    result = text
    
    if mode == 'replace':
        # Thay thế các ký tự đặc biệt bằng từ ngữ thông thường
        for emoji, replacement in char_mapping.items():
            result = result.replace(emoji, replacement)
    elif mode == 'remove':
        # Xóa các ký tự đặc biệt
        for emoji in char_mapping.keys():
            result = result.replace(emoji, '')
    
    return result


def remove_special_characters(text):
    """
    Xóa tất cả các ký tự đặc biệt khỏi văn bản
    
    Args:
        text (str): Văn bản đầu vào
    
    Returns:
        str: Văn bản đã xóa ký tự đặc biệt
    """
    import re
    # Loại bỏ các ký tự không phải chữ cái, số hoặc khoảng trắng
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    # Loại bỏ khoảng trắng dư thừa
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text


def replace_emojis_with_text(text):
    """
    Chỉ thay thế các emoji bằng văn bản mô tả
    
    Args:
        text (str): Văn bản đầu vào
    
    Returns:
        str: Văn bản với emoji được thay thế bằng văn bản
    """
    # Từ điển ánh xạ các emoji phổ biến thành văn bản
    emoji_mapping = {
        '✅': 'yes',
        '❌': 'no',
        '✔': 'yes',
        '✗': 'no',
        '✓': 'yes',
        '✘': 'no',
        '☀': 'sunny',
        '☁': 'cloudy',
        '☂': 'rainy',
        '☃': 'snowy',
        '⚡': 'lightning',
        '❤': 'love',
        '🔥': 'fire',
        '🌟': 'star',
        '💯': 'perfect',
        '🎉': 'celebration',
        '👏': 'clap',
        '🙌': 'raise_hands',
        '👍': 'thumbs_up',
        '👎': 'thumbs_down',
        '👌': 'ok',
        '🙏': 'pray',
        '👀': 'eyes',
        '💡': 'idea',
        '⚠': 'warning',
        '❗': 'exclamation',
        '❓': 'question',
        '❕': 'white_exclamation',
        '❔': 'white_question',
        '❗️': 'exclamation',
        '❓️': 'question',
        '➕': 'plus',
        '➖': 'minus',
        '➗': 'divide',
        '✖': 'multiply',
        '♠': 'spade',
        '♣': 'club',
        '♥': 'heart',
        '♦': 'diamond',
        '💮': 'white_flower',
        '💯': 'hundred_points',
        '✔️': 'check_mark',
        '❌️': 'cross_mark',
        '❗️': 'heavy_exclamation',
        '❓️': 'question_mark',
        '‼️': 'double_exclamation',
        '⁉️': 'exclamation_question',
        '🔅': 'dim_button',
        '🔆': 'bright_button',
        '〽️': 'part_alternation',
        '⚠️': 'warning_sign',
        '🚸': 'children_crossing',
        '⛔': 'no_entry',
        '🚫': 'prohibited',
        '🚳': 'no_bicycles',
        '🚭': 'no_smoking',
        '🚯': 'no_littering',
        '🚱': 'non-potable_water',
        '🚷': 'no_pedestrians',
        '📵': 'no_mobile_phones',
        '🔞': 'underage',
        '☢️': 'radioactive',
        '☣️': 'biohazard',
        '⬆️': 'arrow_up',
        '↗️': 'arrow_up_right',
        '➡️': 'arrow_right',
        '↘️': 'arrow_down_right',
        '⬇️': 'arrow_down',
        '↙️': 'arrow_down_left',
        '⬅️': 'arrow_left',
        '↖️': 'arrow_up_left',
        '↕️': 'arrow_up_down',
        '↔️': 'left_right_arrow',
        '↩️': 'right_arrow_curving_left',
        '↪️': 'left_arrow_curving_right',
        '⤴️': 'right_arrow_curving_up',
        '⤵️': 'right_arrow_curving_down',
        '🔃': 'clockwise_vertical_arrows',
        '🔄': 'counterclockwise_arrows_button',
        '🔙': 'back_arrow',
        '🔚': 'end_arrow',
        '🔛': 'on_arrow',
        '🔜': 'soon_arrow',
        '🔝': 'top_arrow',
        '🛐': 'place_of_worship',
        '⚛️': 'atom_symbol',
        '🕉️': 'om_symbol',
        '✡️': 'star_of_david',
        '☸️': 'wheel_of_dharma',
        '☯️': 'yin_yang',
        '✝️': 'latin_cross',
        '☦️': 'orthodox_cross',
        '☪️': 'star_and_crescent',
        '☮️': 'peace_symbol',
        '🕎': 'menorah',
        '🔯': 'dotted_six-pointed_star',
        '♈': 'aries',
        '♉': 'taurus',
        '♊': 'gemini',
        '♋': 'cancer',
        '♌': 'leo',
        '♍': 'virgo',
        '♎': 'libra',
        '♏': 'scorpio',
        '♐': 'sagittarius',
        '♑': 'capricorn',
        '♒': 'aquarius',
        '♓': 'pisces',
        '⛎': 'ophiuchus',
        '🔀': 'shuffle_tracks',
        '🔁': 'repeat',
        '🔂': 'repeat_single',
        '▶️': 'play_button',
        '⏩': 'fast_forward',
        '⏭️': 'next_track',
        '⏯️': 'play_or_pause',
        '◀️': 'reverse_button',
        '⏪': 'fast_reverse',
        '⏮️': 'last_track',
        '🔼': 'up_button',
        '⏫': 'fast_up',
        '🔽': 'down_button',
        '⏬': 'fast_down',
        '⏸️': 'pause_button',
        '⏹️': 'stop_button',
        '⏺️': 'record_button',
        ' eject_button': 'eject_button',  # Lỗi đánh máy cần sửa
        '🎦': 'cinema',
        '🔅': 'dim_button',
        '🔆': 'bright_button',
        '📶': 'antenna_bars',
        '📳': 'vibration_mode',
        '📴': 'mobile_phone_off',
        '♀️': 'female_sign',
        '♂️': 'male_sign',
        ' transgender_symbol': 'transgender_symbol',  # Lỗi đánh máy cần sửa
        '✖️': 'multiplication_sign',
        '➕': 'plus_sign',
        '➖': 'minus_sign',
        '➗': 'division_sign',
        '♾️': 'infinity',
        '‼️': 'double_exclamation',
        '⁉️': 'exclamation_question',
        '❓️': 'red_question_mark',
        '❔️': 'white_question_mark',
        '❕️': 'white_exclamation_mark',
        '❗️': 'red_exclamation_mark',
        '〰️': 'wavy_dash',
        '💱': 'currency_exchange',
        '💲': 'heavy_dollar_sign',
        '⚕️': 'medical_symbol',
        '♻️': 'recycling_symbol',
        '⚜️': 'fleur_de_lis',
        '🔱': 'trident_emblem',
        '📛': 'name_badge',
        '🔰': 'beginner',
        '⭕': 'hollow_red_circle',
        '✅': 'check_mark_button',
        '☑️': 'check_box_with_check',
        '✔️': 'check_mark',
        '❌': 'cross_mark',
        '❎': 'cross_mark_button',
        '➰': 'curly_loop',
        '➿': 'double_curly_loop',
        '〽️': 'part_alternation_mark',
        '✳️': 'eight_spoked_asterisk',
        '✴️': 'eight_pointed_star',
        '❇️': 'sparkle',
        '©️': 'copyright',
        '®️': 'registered',
        '™️': 'trade_mark',
        '🔠': 'latin_uppercase',
        '🔡': 'latin_lowercase',
        '🔢': 'numbers',
        '🔣': 'symbols',
        '🔤': 'latin_letters',
        '🅰️': 'a_button',
        '🆎': 'ab_button',
        '🅱️': 'b_button',
        '🆑': 'cl_button',
        '🆒': 'cool_button',
        '🆓': 'free_button',
        'ℹ️': 'information',
        '🆔': 'id_button',
        'Ⓜ️': 'circled_m',
        '🆕': 'new_button',
        '🆖': 'ng_button',
        '🅾️': 'o_button',
        '🆗': 'ok_button',
        '🅿️': 'parking',
        '🆘': 'sos_button',
        '🆙': 'up_button',
        '🆚': 'vs_button',
        '🈁': 'koko_button',
        '🈂️': 'sa_button',
        '🈷️': 'monthly_amount',
        '🈶': 'not_free',
        '🈯': 'reserved',
        '🉐': 'bargain',
        '🈹': 'discount',
        '🈚': 'free',
        '🈲': 'prohibited',
        '🉑': 'acceptable',
        '🈸': 'application',
        '🈴': 'passing_grade',
        '🈳': 'vacancy',
        '㊗️': 'congratulations',
        '㊙️': 'secret',
        '🈺': 'open_business',
        '🈵': 'full',
        '🔴': 'red_circle',
        '🟠': 'orange_circle',
        '🟡': 'yellow_circle',
        '🟢': 'green_circle',
        '🔵': 'blue_circle',
        '🟣': 'purple_circle',
        '🟤': 'brown_circle',
        '⚫': 'black_circle',
        '⚪': 'white_circle',
        '🟥': 'red_square',
        '🟧': 'orange_square',
        '🟨': 'yellow_square',
        '🟩': 'green_square',
        '🟦': 'blue_square',
        '🟪': 'purple_square',
        '🟫': 'brown_square',
        '⬛': 'black_large_square',
        '⬜': 'white_large_square',
        '◼️': 'black_medium_square',
        '◻️': 'white_medium_square',
        '◾': 'black_medium_small_square',
        '◽': 'white_medium_small_square',
        '▪️': 'black_small_square',
        '▫️': 'white_small_square',
        '🔶': 'large_orange_diamond',
        '🔷': 'large_blue_diamond',
        '🔸': 'small_orange_diamond',
        '🔹': 'small_blue_diamond',
        '🔺': 'red_triangle',
        '🔻': 'down_red_triangle',
        '💠': 'diamond_with_dot',
        '🔘': 'radio_button',
        '🔳': 'white_square_button',
        '🔲': 'black_square_button',
        '😊': 'smiling_face',
        '😌': 'relieved_face',
        '😍': 'heart_eyes',
        '😏': 'smirk',
        '😒': 'unamused',
        '😞': 'disappointed',
        '😔': 'pensive',
        '😟': 'worried',
        '😕': 'confused',
        '🙁': 'slightly_frowning',
        '☹': 'frowning',
        '😮': 'open_mouth',
        '😯': 'hushed',
        '😲': 'astonished',
        '😳': 'flushed',
        '🥺': 'pleading',
        '😦': 'frowning_open_mouth',
        '😧': 'anguished',
        '😨': 'fearful',
        '😰': 'cold_sweat',
        '😥': 'disappointed_relieved',
        '😢': 'crying',
        '😭': 'loudly_crying',
        '😱': 'screaming',
        '😖': 'confounded',
        '😣': 'persevering',
        '😞': 'disappointed',
        '😓': 'cold_sweat',
        '😩': 'weary',
        '😫': 'tired',
        '🥱': 'yawning',
        '😤': 'triumph',
        '😡': 'angry',
        '😠': 'angry',
        '🤬': 'cursing',
        '😈': 'devil',
        '👿': 'angry_devil',
        '💀': 'skull',
        '☠': 'skull_crossbones',
        '💩': 'poop',
        '🤡': 'clown',
        '👹': 'japanese_ogre',
        '👺': 'japanese_goblin',
        '👻': 'ghost',
        '👽': 'alien',
        '👾': 'space_invader',
        '🤖': 'robot',
        '😺': 'cat',
        '😸': 'grinning_cat',
        '😹': 'tears_of_joy_cat',
        '😻': 'heart_eyes_cat',
        '😼': 'wry_smile_cat',
        '😽': 'kissing_cat',
        '🙀': 'weary_cat',
        '😿': 'crying_cat',
        '😾': 'pouting_cat',
        '🙈': 'see_no_evil',
        '🙉': 'hear_no_evil',
        '🙊': 'speak_no_evil',
    }
    
    result = text
    for emoji, replacement in emoji_mapping.items():
        result = result.replace(emoji, replacement)
    
    return result


# Hàm tiện ích để xử lý các trường hợp cụ thể
def process_special_chars(text, replacements=None, mode='replace'):
    """
    Hàm tổng quát để xử lý các ký tự đặc biệt
    
    Args:
        text (str): Văn bản đầu vào
        replacements (dict): Từ điển thay thế tùy chỉnh (nếu có)
        mode (str): Chế độ - 'replace' hoặc 'remove'
    
    Returns:
        str: Văn bản đã xử lý
    """
    if replacements is None:
        # Sử dụng từ điển mặc định
        replacements = {
            '✅': 'yes',
            '❌': 'no',
            '✔': 'yes',
            '✗': 'no',
            '✓': 'yes',
            '✘': 'no',
        }
    
    result = text
    
    if mode == 'replace':
        for char, replacement in replacements.items():
            result = result.replace(char, replacement)
    elif mode == 'remove':
        for char in replacements.keys():
            result = result.replace(char, '')
    
    return result


# Ví dụ sử dụng
if __name__ == "__main__":
    # Ví dụ 1: Thay thế ký tự đặc biệt
    text1 = "Nhiệm vụ này ✅ hoàn thành, nhưng nhiệm vụ kia ❌ thất bại"
    print("Văn bản gốc:", text1)
    print("Sau khi thay thế:", replace_special_characters(text1))
    print()
    
    # Ví dụ 2: Xóa ký tự đặc biệt
    text2 = "Ký hiệu: ✅❌✔✗"
    print("Văn bản gốc:", text2)
    print("Sau khi xóa:", replace_special_characters(text2, mode='remove'))
    print()
    
    # Ví dụ 3: Chỉ thay thế emoji
    text3 = "Tôi rất vui 😊 nhưng cũng có chút buồn 😢"
    print("Văn bản gốc:", text3)
    print("Sau khi thay thế emoji:", replace_emojis_with_text(text3))