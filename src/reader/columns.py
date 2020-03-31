class_dict = {
    '1': {
        'name': "postać nie T i nie B",
    },
    '2': {
        'name': "postać T",
    },
    '3': {
        'name': "postać T",
    },
    '4': {
        'name': "mielolastyczna o niskim zróznicowaniu",
    },
    '5': {
        'name': "mielolastyczna z dojrzewaniem",
    },
    '6': {},
    '7': {},
    '8': {},
    '9': {},
    '10': {},
    '11': {},
    '12': {},
    '13': {},
    '14': {},
    '15': {},
    '16': {},
    '17': {},
    '18': {},
    '19': {},
    '20': {},
}

symptoms_dict = {
    '1': {
        'name': "Temperatura",
        'values': {
            1: "tak",
            2: "nie"
        }
    },
    '2': {
        'name': "Anemia",
        'values': {
            1: "średnia",
            2: "średnio ciężka",
            3: "ciężka"
        }
    },
    '3': {
        'name': "Stopień krwawienia",
        'values': {
            1: "mały",
            2: "duży",
        }
    },
    '4': {
        'name': "Miejsce krwawienia",
        'values': {
            1: 'skóra',
            2: 'jama ustna',
            3: 'jama nosowa',
            4: 'krwawienie do siatkówki oka',
            5: 'drogi oddechowe',
            6: 'przewód moczowy',
            7: 'przewód trawienny',
            8: 'mózg'
        }
    },
    '5': {
        'name': "Bóle kości",
        'values': {
            1: 'tak',
            2: 'nie'
        }
    },
    '6': {
        'name': "Wrażliwość mostka",
        'values': {
            1: 'tak',
            2: 'nie'
        }
    },
    '7': {
        'name': "Powiększenie węzłów chłonnych",
        'values': {
            1: 'nieznaczne',
            2: 'silne'
        }
    },
    '8': {
        'name': "Powiększnie wątroby i śledziony",
        'values': {
            1: 'nieznaczne',
            2: 'silne'
        }
    },
    '9': {
        'name': "Centralny układ nerwowy (ból głowy, wymioty, drgawki, senność, śpiączka)",
        'values': {
            1: 'tak',
            2: 'nie'
        }
    },
    '10': {
        'name': 'Powiększenie jąder',
        'values': {
            1: 'tak',
            2: 'nie'
        }
    },
    '11': {
        'name': 'Uszkodzenie w sercu, płucach, nerce',
        'values': {
            1: 'tak',
            2: 'nie'
        }
    },
    '12': {
        'name': 'Gałka oczna, zaburzenia widzenia, krwiawienie do siatkówki, wytrzeszcz oczu',
        'values': {
            1: 'tak',
            2: 'nie'
        }
    },
    '13': {
        'name': 'Poziom WBC (leukocytów)',
        'values': {
            1: 'powiększony',
            2: 'obniżony',
            3: 'normalny'
        }
    },
    '14': {
        'name': 'Obniżenie liczby RBC (erytrocytów?)',
        'values': {
            1: 'lekkie',
            2: 'średnie',
            3: 'duże'
        }
    },
    '15': {
        'name': 'Liczba płytek krwi',
        'values': {
            1: 'obniżona',
            2: 'normalna',
        }
    },
    '16': {
        'name': 'Niedojrzałe komórki (blastyczne?)',
        'values': {
            1: 'istnieją',
            2: 'nie istnieją',
        }
    },
    '17': {
        'name': 'Stan pobudzenia szpiku',
        'values': {
            1: 'krańcowo czynny',
            2: 'średnio czynny',
            3: 'czynny'
        }
    },
    '18': {
        'name': 'Główne komórki w szpiku?',
        "values": {
            1: 'prymitywne i niedojrzałe',
            2: "wcześniej niedojrzałe granulocyty",
            3: "dojrzałe"
        }
    },
    '19': {
        'name': "Poziom limfocytów",
        "values": {
            1: "duży",
            2: "niski",
            3: "nieregularny"
        }
    },
    '20': {
        'name': "Reakcja (Test chemiczny tkanki)",
        "values": {
            1: "pozytywna",
            2: "negatywna"
        }
    }
}

symptoms_name_dict = {key: value['name'] for key, value in symptoms_dict.items()}

