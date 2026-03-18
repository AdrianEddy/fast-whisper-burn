use serde::ser::StdError;
use std::{collections::HashSet, fmt, result};
use strum::IntoEnumIterator;

use tokenizers::AddedToken;

pub type Result<T> = result::Result<T, Box<dyn StdError + Send + Sync + 'static>>;

pub struct Gpt2Tokenizer {
    tokenizer: tokenizers::Tokenizer,
    special_token_ids: HashSet<usize>,
}

impl Gpt2Tokenizer {
    pub fn new(file_path: &str) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(file_path)?;

        let special_token_ids = tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .filter_map(|(token_id, token)| token.special.then_some(token_id as usize))
            .collect();

        Ok(Self {
            tokenizer,
            special_token_ids,
        })
    }
    pub fn new_from_data(data: impl AsRef<[u8]>) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_bytes(data)?;

        let special_token_ids = tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .filter_map(|(token_id, token)| token.special.then_some(token_id as usize))
            .collect();

        Ok(Self {
            tokenizer,
            special_token_ids,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let tokens = self.tokenizer.encode(text, false).unwrap();
        tokens.get_ids().iter().map(|t| *t as usize).collect()
    }

    pub fn special_token(&self, token: SpecialToken) -> Option<usize> {
        token
            .token_candidates()
            .into_iter()
            .find_map(|token| self.tokenizer.token_to_id(&token).map(|t| t as usize))
    }

    pub fn decode(&self, tokens: &[usize], skip_special: bool) -> Result<String> {
        self.tokenizer.decode(
            &tokens.iter().map(|t| *t as u32).collect::<Vec<u32>>(),
            skip_special,
        )
    }

    pub fn is_special(&self, token: usize) -> bool {
        self.special_token_ids.contains(&token)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Returns true if the model is multilingual (n_vocab >= 51865).
    /// English-only models have n_vocab = 51864.
    pub fn is_multilingual(&self) -> bool {
        self.vocab_size() >= 51865
    }
}

pub const LANGUAGES: [&str; 100] = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
];

use strum_macros::EnumIter;

#[derive(Debug, Copy, Clone, EnumIter)]
pub enum Language {
    English,
    Chinese,
    German,
    Spanish,
    Russian,
    Korean,
    French,
    Japanese,
    Portuguese,
    Turkish,
    Polish,
    Catalan,
    Dutch,
    Arabic,
    Swedish,
    Italian,
    Indonesian,
    Hindi,
    Finnish,
    Vietnamese,
    Hebrew,
    Ukrainian,
    Greek,
    Malay,
    Czech,
    Romanian,
    Danish,
    Hungarian,
    Tamil,
    Norwegian,
    Thai,
    Urdu,
    Croatian,
    Bulgarian,
    Lithuanian,
    Latin,
    Maori,
    Malayalam,
    Welsh,
    Slovak,
    Telugu,
    Persian,
    Latvian,
    Bengali,
    Serbian,
    Azerbaijani,
    Slovenian,
    Kannada,
    Estonian,
    Macedonian,
    Breton,
    Basque,
    Icelandic,
    Armenian,
    Nepali,
    Mongolian,
    Bosnian,
    Kazakh,
    Albanian,
    Swahili,
    Galician,
    Marathi,
    Punjabi,
    Sinhala,
    Khmer,
    Shona,
    Yoruba,
    Somali,
    Afrikaans,
    Occitan,
    Georgian,
    Belarusian,
    Tajik,
    Sindhi,
    Gujarati,
    Amharic,
    Yiddish,
    Lao,
    Uzbek,
    Faroese,
    HaitianCreole,
    Pashto,
    Turkmen,
    Nynorsk,
    Maltese,
    Sanskrit,
    Luxembourgish,
    Burmese,
    Tibetan,
    Tagalog,
    Malagasy,
    Assamese,
    Tatar,
    Hawaiian,
    Lingala,
    Hausa,
    Bashkir,
    Javanese,
    Sundanese,
    Cantonese,
}

impl Language {
    pub fn as_str(&self) -> &str {
        match self {
            Language::English => "en",
            Language::Chinese => "zh",
            Language::German => "de",
            Language::Spanish => "es",
            Language::Russian => "ru",
            Language::Korean => "ko",
            Language::French => "fr",
            Language::Japanese => "ja",
            Language::Portuguese => "pt",
            Language::Turkish => "tr",
            Language::Polish => "pl",
            Language::Catalan => "ca",
            Language::Dutch => "nl",
            Language::Arabic => "ar",
            Language::Swedish => "sv",
            Language::Italian => "it",
            Language::Indonesian => "id",
            Language::Hindi => "hi",
            Language::Finnish => "fi",
            Language::Vietnamese => "vi",
            Language::Hebrew => "he",
            Language::Ukrainian => "uk",
            Language::Greek => "el",
            Language::Malay => "ms",
            Language::Czech => "cs",
            Language::Romanian => "ro",
            Language::Danish => "da",
            Language::Hungarian => "hu",
            Language::Tamil => "ta",
            Language::Norwegian => "no",
            Language::Thai => "th",
            Language::Urdu => "ur",
            Language::Croatian => "hr",
            Language::Bulgarian => "bg",
            Language::Lithuanian => "lt",
            Language::Latin => "la",
            Language::Maori => "mi",
            Language::Malayalam => "ml",
            Language::Welsh => "cy",
            Language::Slovak => "sk",
            Language::Telugu => "te",
            Language::Persian => "fa",
            Language::Latvian => "lv",
            Language::Bengali => "bn",
            Language::Serbian => "sr",
            Language::Azerbaijani => "az",
            Language::Slovenian => "sl",
            Language::Kannada => "kn",
            Language::Estonian => "et",
            Language::Macedonian => "mk",
            Language::Breton => "br",
            Language::Basque => "eu",
            Language::Icelandic => "is",
            Language::Armenian => "hy",
            Language::Nepali => "ne",
            Language::Mongolian => "mn",
            Language::Bosnian => "bs",
            Language::Kazakh => "kk",
            Language::Albanian => "sq",
            Language::Swahili => "sw",
            Language::Galician => "gl",
            Language::Marathi => "mr",
            Language::Punjabi => "pa",
            Language::Sinhala => "si",
            Language::Khmer => "km",
            Language::Shona => "sn",
            Language::Yoruba => "yo",
            Language::Somali => "so",
            Language::Afrikaans => "af",
            Language::Occitan => "oc",
            Language::Georgian => "ka",
            Language::Belarusian => "be",
            Language::Tajik => "tg",
            Language::Sindhi => "sd",
            Language::Gujarati => "gu",
            Language::Amharic => "am",
            Language::Yiddish => "yi",
            Language::Lao => "lo",
            Language::Uzbek => "uz",
            Language::Faroese => "fo",
            Language::HaitianCreole => "ht",
            Language::Pashto => "ps",
            Language::Turkmen => "tk",
            Language::Nynorsk => "nn",
            Language::Maltese => "mt",
            Language::Sanskrit => "sa",
            Language::Luxembourgish => "lb",
            Language::Burmese => "my",
            Language::Tibetan => "bo",
            Language::Tagalog => "tl",
            Language::Malagasy => "mg",
            Language::Assamese => "as",
            Language::Tatar => "tt",
            Language::Hawaiian => "haw",
            Language::Lingala => "ln",
            Language::Hausa => "ha",
            Language::Bashkir => "ba",
            Language::Javanese => "jw",
            Language::Sundanese => "su",
            Language::Cantonese => "yue",
        }
    }

    pub fn from_code(code: &str) -> Option<Self> {
        Self::iter().find(|language| language.as_str() == code)
    }
}

pub enum SpecialToken {
    EndofText,
    StartofTranscript,
    Translate,
    Transcribe,
    StartofLM,
    StartofPrev,
    NoSpeech,
    NoTimeStamps,
    Language(Language),
    Timestamp(f64),
}

impl SpecialToken {
    fn token_candidates(&self) -> Vec<String> {
        match self {
            SpecialToken::EndofText => vec!["<|endoftext|>".to_owned()],
            SpecialToken::StartofTranscript => vec!["<|startoftranscript|>".to_owned()],
            SpecialToken::Translate => vec!["<|translate|>".to_owned()],
            SpecialToken::Transcribe => vec!["<|transcribe|>".to_owned()],
            SpecialToken::StartofLM => vec!["<|startoflm|>".to_owned()],
            SpecialToken::StartofPrev => vec!["<|startofprev|>".to_owned()],
            SpecialToken::NoSpeech => {
                vec!["<|nocaptions|>".to_owned(), "<|nospeech|>".to_owned()]
            }
            SpecialToken::NoTimeStamps => vec!["<|notimestamps|>".to_owned()],
            SpecialToken::Language(lang) => vec![lang.special_token().to_owned()],
            SpecialToken::Timestamp(val) => vec![timestamp_token(*val)],
        }
    }
}

impl fmt::Display for SpecialToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpecialToken::EndofText => write!(f, "<|endoftext|>"),
            SpecialToken::StartofTranscript => write!(f, "<|startoftranscript|>"),
            SpecialToken::Translate => write!(f, "<|translate|>"),
            SpecialToken::Transcribe => write!(f, "<|transcribe|>"),
            SpecialToken::StartofLM => write!(f, "<|startoflm|>"),
            SpecialToken::StartofPrev => write!(f, "<|startofprev|>"),
            SpecialToken::NoSpeech => write!(f, "<|nocaptions|>"),
            SpecialToken::NoTimeStamps => write!(f, "<|notimestamps|>"),
            SpecialToken::Language(lang) => write!(f, "{}", lang.special_token()),
            SpecialToken::Timestamp(val) => write!(f, "{}", timestamp_token(*val)),
        }
    }
}

impl Language {
    fn special_token(&self) -> &'static str {
        match self {
            Language::English => "<|en|>",
            Language::Chinese => "<|zh|>",
            Language::German => "<|de|>",
            Language::Spanish => "<|es|>",
            Language::Russian => "<|ru|>",
            Language::Korean => "<|ko|>",
            Language::French => "<|fr|>",
            Language::Japanese => "<|ja|>",
            Language::Portuguese => "<|pt|>",
            Language::Turkish => "<|tr|>",
            Language::Polish => "<|pl|>",
            Language::Catalan => "<|ca|>",
            Language::Dutch => "<|nl|>",
            Language::Arabic => "<|ar|>",
            Language::Swedish => "<|sv|>",
            Language::Italian => "<|it|>",
            Language::Indonesian => "<|id|>",
            Language::Hindi => "<|hi|>",
            Language::Finnish => "<|fi|>",
            Language::Vietnamese => "<|vi|>",
            Language::Hebrew => "<|he|>",
            Language::Ukrainian => "<|uk|>",
            Language::Greek => "<|el|>",
            Language::Malay => "<|ms|>",
            Language::Czech => "<|cs|>",
            Language::Romanian => "<|ro|>",
            Language::Danish => "<|da|>",
            Language::Hungarian => "<|hu|>",
            Language::Tamil => "<|ta|>",
            Language::Norwegian => "<|no|>",
            Language::Thai => "<|th|>",
            Language::Urdu => "<|ur|>",
            Language::Croatian => "<|hr|>",
            Language::Bulgarian => "<|bg|>",
            Language::Lithuanian => "<|lt|>",
            Language::Latin => "<|la|>",
            Language::Maori => "<|mi|>",
            Language::Malayalam => "<|ml|>",
            Language::Welsh => "<|cy|>",
            Language::Slovak => "<|sk|>",
            Language::Telugu => "<|te|>",
            Language::Persian => "<|fa|>",
            Language::Latvian => "<|lv|>",
            Language::Bengali => "<|bn|>",
            Language::Serbian => "<|sr|>",
            Language::Azerbaijani => "<|az|>",
            Language::Slovenian => "<|sl|>",
            Language::Kannada => "<|kn|>",
            Language::Estonian => "<|et|>",
            Language::Macedonian => "<|mk|>",
            Language::Breton => "<|br|>",
            Language::Basque => "<|eu|>",
            Language::Icelandic => "<|is|>",
            Language::Armenian => "<|hy|>",
            Language::Nepali => "<|ne|>",
            Language::Mongolian => "<|mn|>",
            Language::Bosnian => "<|bs|>",
            Language::Kazakh => "<|kk|>",
            Language::Albanian => "<|sq|>",
            Language::Swahili => "<|sw|>",
            Language::Galician => "<|gl|>",
            Language::Marathi => "<|mr|>",
            Language::Punjabi => "<|pa|>",
            Language::Sinhala => "<|si|>",
            Language::Khmer => "<|km|>",
            Language::Shona => "<|sn|>",
            Language::Yoruba => "<|yo|>",
            Language::Somali => "<|so|>",
            Language::Afrikaans => "<|af|>",
            Language::Occitan => "<|oc|>",
            Language::Georgian => "<|ka|>",
            Language::Belarusian => "<|be|>",
            Language::Tajik => "<|tg|>",
            Language::Sindhi => "<|sd|>",
            Language::Gujarati => "<|gu|>",
            Language::Amharic => "<|am|>",
            Language::Yiddish => "<|yi|>",
            Language::Lao => "<|lo|>",
            Language::Uzbek => "<|uz|>",
            Language::Faroese => "<|fo|>",
            Language::HaitianCreole => "<|ht|>",
            Language::Pashto => "<|ps|>",
            Language::Turkmen => "<|tk|>",
            Language::Nynorsk => "<|nn|>",
            Language::Maltese => "<|mt|>",
            Language::Sanskrit => "<|sa|>",
            Language::Luxembourgish => "<|lb|>",
            Language::Burmese => "<|my|>",
            Language::Tibetan => "<|bo|>",
            Language::Tagalog => "<|tl|>",
            Language::Malagasy => "<|mg|>",
            Language::Assamese => "<|as|>",
            Language::Tatar => "<|tt|>",
            Language::Hawaiian => "<|haw|>",
            Language::Lingala => "<|ln|>",
            Language::Hausa => "<|ha|>",
            Language::Bashkir => "<|ba|>",
            Language::Javanese => "<|jw|>",
            Language::Sundanese => "<|su|>",
            Language::Cantonese => "<|yue|>",
        }
    }
}

fn timestamp_token(value: f64) -> String {
    format!("<|{value:.2}|>")
}

fn _construct_special_tokens() -> Vec<AddedToken> {
    const SPEC1: [&str; 2] = ["<|endoftext|>", "<|startoftranscript|>"];

    let lang_keys = LANGUAGES.iter().map(|lang| format!("<|{lang}|>"));

    const SPEC2: [&str; 6] = [
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nocaptions|>",
        "<|notimestamps|>",
    ];

    let range_keys = (0..1501)
        .map(|i| i as f64 * 0.02)
        .map(|f| format!("<|{f:.2}|>"));

    SPEC1
        .into_iter()
        .map(String::from)
        .chain(lang_keys)
        .chain(SPEC2.into_iter().map(String::from))
        .chain(range_keys)
        .map(|tok| AddedToken::from(tok, true))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{Gpt2Tokenizer, SpecialToken};

    #[test]
    fn encode_does_not_inject_template_tokens() {
        let tokenizer = Gpt2Tokenizer::new("base_en").unwrap();

        let tokens = tokenizer.encode("hello world");

        assert!(!tokens.is_empty());
        assert_ne!(
            tokens[0],
            tokenizer
                .special_token(SpecialToken::StartofTranscript)
                .unwrap()
        );
        assert!(!tokens.contains(&tokenizer.special_token(SpecialToken::NoTimeStamps).unwrap()));
        assert_ne!(
            tokens[tokens.len() - 1],
            tokenizer.special_token(SpecialToken::EndofText).unwrap()
        );
    }

    #[test]
    fn no_speech_token_uses_whisper_cpp_slot() {
        let tokenizer = Gpt2Tokenizer::new("base_en").unwrap();

        assert_eq!(tokenizer.special_token(SpecialToken::NoSpeech), Some(50361));
        assert!(tokenizer.is_special(50361));
    }
}
