import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to your converted model
model_path = "/scratch/vladimir_albrekht/projects/smollm/sft/output/nanotron_500M_126000_hf_sft3_aspd_data/checkpoint-139"
# (prompt, max_new_tokens=100, temperature=0.1, top_p=0.9, repetition_penalty=1.0, top_k=50)
# model_path = "/scratch/vladimir_albrekht/projects/smollm/output/nanotron_500M/126000/hf"
# (prompt, max_new_tokens=100, temperature=0.1, top_p=0.9, repetition_penalty=2.0, top_k=50)

FILE_SAMPLES_PATH = "/scratch/vladimir_albrekht/projects/smollm/data/sft_aspandiyar_converted/instruction_input/converted_with_instruction_tags.json"

def load_prompts_from_file(file_path, limit=None):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = []
    for item in data:
        if "messages" in item:
            for msg in item["messages"]:
                if msg["role"] == "user":
                    prompts.append(msg["content"])
                    break  # take only the first user message per conversation

    if limit:
        prompts = prompts[:limit]

    return prompts
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
    device_map="auto"  # Automatically determine device placement
)

# Set the model to evaluation mode
model.eval()

# Function to generate text
def generate_text(prompt, max_new_tokens=100, temperature=0.1, top_p=0.9, repetition_penalty=1.0, top_k=50, base_model=False):
    if hasattr(tokenizer, "apply_chat_template") and base_model == False:
        messages = [
            # {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # print(f"\nFormatted prompt:\n{prompt_text}\n")
    else:
        prompt_text = prompt  # fallback to plain text

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            top_k=top_k
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


# Example usage
if __name__ == "__main__":
    # prompts = load_prompts_from_file(FILE_SAMPLES_PATH, limit=1)
    # print("Model loaded successfully! Generating text samples...\n")

    # for prompt in prompts:
    #     print(f"Prompt: {prompt}")
    #     result = generate_text(prompt)
    #     print(f"Generated: {result}\n")
    #     print("-" * 80 + "\n")
    # Test the model with a few prompts
    prompts = [
        "<|@instruction@|>Абай Кунанбавев великий писатель казахского народа</|@instruction@|> <|@question@|>Абай Құнанбаев кім болған? </|@question@|>",
        # "Radio City — Үндістанның алғашқы жеке FM радиостанциясы және 2001 жылдың 3 шілдесінде іске қосылды. Ол Мумбайдан (2004 жылы басталған), Бенгалурудан (алғаш рет 2001 жылы басталды), Лакхнаудан және 91,1 (көптеген қалаларда бұрын 91,0) мегагерцте таратады. Нью-Дели (2003 жылдан бастап). Ол хинди, ағылшын және аймақтық әндерді ойнайды. Ол Хайдарабадта 2006 жылдың наурызында, Ченнайда 2006 жылдың 7 шілдесінде және Вишакхапатнамда 2007 жылдың қазанында іске қосылды. Радио Сити жақында 2008 жылдың мамыр айында музыкаға қатысты жаңалықтарды, бейнелерді, бейнелерді ұсынатын PlanetRadiocity.com музыкалық порталын іске қосу арқылы Жаңа медиаға шықты. әндер және басқа музыкаға қатысты мүмкіндіктер. Радиостанция қазіргі уақытта хинди және аймақтық музыканың қоспасын ойнайды. Авраам Томас компанияның бас директоры болып табылады. Албанияда футбол Албания футбол федерациясы (FSHF) құрылғанға дейін болған. Бұған команданың 1929-1931 жылдардағы Балқан кубогы турниріне тіркелуі дәлел болды, ол 1929 жылы басталған (Албания ақырында бәсекелестікке байланысты командалардың қысымына ұшырағанымен, бәсекелестік бірінші басталып, жекпе-жектерде жеткілікті күшті болды). Албания құрамасы 1930 жылы 6 маусымда құрылды, бірақ Албания өзінің алғашқы халықаралық матчын өткізу үшін 16 жыл күтуге мәжбүр болды, содан кейін 1946 жылы Югославияны жеңді. 1932 жылы Албания ФИФА-ға қосылды (12-16 маусымда өткен конвенция кезінде) Ал 1954 жылы ол УЕФА-ның негізін қалаушылардың бірі болды. Echosmith - 2009 жылдың ақпанында Калифорния штатының Чино қаласында құрылған американдық корпоративтік инди-поп тобы. Бастапқыда бауырластардың квартеті ретінде құрылған топ қазіргі уақытта 2016 жылдың аяғында үлкен ағасы Джейми кеткеннен кейін Сидней, Ноа және Грэм Сиеротадан тұрады. Эхосмит алдымен «Дайын Set Go!» ретінде бастады. 2012 жылдың мамырында Warner Bros. Records-қа қол қойғанға дейін. Олар \'Billboard\' Hot 100 тізімінде 13-ші орынға жеткен және RIAA-да 1 200 000-нан астам сатылыммен қос платина сертификатына ие болған \'Cool Kids\' хит әнімен танымал. Америка Құрама Штаттары, сонымен қатар Австралиядағы ARIA қос платина. Ән Warner Bros. Records-тың 2014 жылғы ең көп сатылған цифрлық әні болды, ол 1,3 миллион жүктеп алынған. Топтың «Talking Dreams» атты дебюттік альбомы 2013 жылдың 8 қазанында жарық көрді. Оңтүстік Америка Құрама Штаттарындағы әйелдер колледждері студенттер қауымы тек қана немесе тек әйелдерден тұратын бакалавриат, бакалавр дәрежесін беретін оқу орындарын, көбінесе гуманитарлық колледждерді білдіреді. , Америка Құрама Штаттарының оңтүстігінде орналасқан. Көпшілігі алдымен қыздар семинариясы немесе академиясы ретінде басталды. Салем колледжі - Оңтүстіктегі ең көне әйелдер оқу орны және Уэслиан колледжі әйелдерге арналған колледж ретінде арнайы құрылған алғашқы колледж. Мэри Болдуин университеті және Салем колледжі сияқты кейбір мектептер магистратура деңгейінде бірлескен білім беру курстарын ұсынады. Бірінші Артур округінің сот ғимараты және түрме, мүмкін, Құрама Штаттардағы ең кішкентай сот үйі болды және қазір мұражай ретінде қызмет етеді. Arthur's Magazine (1844–1846) — 19 ғасырда Филадельфияда шыққан американдық әдеби мерзімді басылым. Т.С.Артур өңдеген, онда Эдгар А.По, Дж.Х.Инграхам, Сара Джозефа Хэйл, Томас Дж.Спир және т.б. 1846 жылы мамырда ол «Годей ханымның кітабына» біріктірілді. Хоккейден 2014–15 жылғы Украина чемпионаты хоккейден Украина чемпионатының 23-ші маусымы болды. Украинадағы тұрақсыздыққа және көптеген клубтардың экономикалық мәселелеріне байланысты осы маусымда лигаға тек төрт команда қатысты. «Генералы Киев» өткен маусымда лигаға қатысқан жалғыз команда болды және маусым 2014 жылдың соңынан кейін басталды. Тұрақты маусым бар болғаны 12 турды қамтыды, онда барлық командалар жартылай финалға өтті. Финалда Киевтің АТЭК командасы тұрақты маусымның жеңімпазы Х.К.Кременчукты жеңді. Әйелдерге арналған бірінші журнал - АҚШ-тағы Bauer Media Group шығаратын әйелдер журналы. Журнал 1989 жылы шыққан. Ол Энглвуд Клиффс, Нью-Джерси қаласында орналасқан. 2011 жылы журналдың таралымы 1 310 696 дананы құрады. Freeway Complex өрті 2008 жылы Калифорния штатындағы Оранж округінің Санта-Ана каньоны аймағындағы орман өрті болды. Өрт 2008 жылдың 15 қарашасында екі бөлек өрт ретінде басталды. «Жолдағы өрт» алдымен таңғы сағат 9-да басталды, «Қоқыс алаңындағы өрт» шамамен 2 сағаттан кейін тұтанды. Бұл екі бөлек өрт бір күннен кейін қосылып, ақырында Анахайм Хиллс пен Йорба Линдадағы 314 резиденцияны қиратты. Уильям Раст - Джастин Тимберлейк пен Трейс Айала негізін қалаған американдық киім желісі. Бұл премиум джинсыларымен танымал. 2006 жылы 17 қазанда Джастин Тимберлейк пен Трейс Аяла Уильям Расттың жаңа киім желісін шығару үшін алғашқы сән көрсетілімін өткізді. Жапсырма сонымен қатар курткалар мен үстер сияқты басқа киімдерді шығарады. Компания алдымен джинсы желісі ретінде басталды, кейінірек ерлер мен әйелдер киімінің желісіне айналды.','input':'Ең бірінші Артур журналы немесе Әйелдерге арналған бірінші журнал қай журналды шығарды?",
        "<|@instruction@|>Салем Qwen!</|@instruction@|> <|@question@|> Қазақстандағы қанша қала бар?</|@question@|>",
        "<|@instruction@|>Азшылық көшбасшысының рөлдері мен жауапкершілігі нақт</|@instruction@|> <|@question@|>Азшылықтардың көшбасшылық рөлдері нақты көрсетілген бе?</|@question@|>",
        "Казакстаннын президенті кім?",
        "ISSAI қай жерде орналасқан?",
        "DanMachi романының авторы кім?",
        "Мен саған бұрынан бері айтым келді, мен сені жақсы көремін"
    ]
    
    print("Model loaded successfully! Generating text samples...\n")
    
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        result = generate_text(prompt)
        print(f"Generated: {result}\n")
        print("-" * 80 + "\n")