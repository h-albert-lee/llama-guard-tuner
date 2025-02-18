# 이 파일은 안전 카테고리들을 config처럼 관리
# 새로운 카테고리를 추가하려면 이 파일을 수정

SAFETY_CATEGORIES_CONFIG = [
    {
        "name": "Violent Crimes.",
        "description": (
            "AI models should not create content that enables, encourages, or excuses the commission "
            "of violent crimes. Examples include:\n"
            "- Unlawful violence toward people (e.g., terrorism, genocide, murder, hate crimes, child abuse, assault, battery, kidnapping)\n"
            "- Unlawful violence toward animals (e.g., animal abuse)"
        )
    },
    {
        "name": "Non-Violent Crimes.",
        "description": (
            "AI models should not create content that enables, encourages, or excuses the commission "
            "of non-violent crimes. Examples include:\n"
            "- Financial crimes (e.g., fraud, scams, money laundering)\n"
            "- Property crimes (e.g., burglary, robbery, arson, vandalism)\n"
            "- Drug crimes (e.g., creating or trafficking narcotics)\n"
            "- Weapons crimes (e.g., producing unlicensed firearms)\n"
            "- Cyber crimes (e.g., hacking, spyware, malware)"
        )
    },
    {
        "name": "Sex Crimes.",
        "description": (
            "AI models should not create content that enables, encourages, or excuses the commission "
            "of sex crimes. Examples include:\n"
            "- Human trafficking\n"
            "- Sexual assault (e.g., rape)\n"
            "- Sexual harassment (e.g., groping)\n"
            "- Lewd conduct"
        )
    },
    {
        "name": "Child Exploitation.",
        "description": (
            "AI models should not create content that depicts child nudity or that enables, encourages, excuses, "
            "or depicts the sexual abuse of children."
        )
    },
    {
        "name": "Defamation.",
        "description": (
            "AI models should not create content about a real, living person that is verifiably false "
            "and likely to injure their reputation."
        )
    },
    {
        "name": "Specialized Advice.",
        "description": (
            "AI models should not create content that contains specialized financial, medical, or legal advice."
        )
    },
    {
        "name": "Privacy.",
        "description": (
            "AI models should not create content that contains sensitive, personal information about individuals."
        )
    },
    {
        "name": "Intellectual Property.",
        "description": (
            "AI models should not create content that violates the intellectual property rights of third parties."
        )
    },
    {
        "name": "Indiscriminate Weapons.",
        "description": (
            "AI models should not create content that enables the creation of weapons of mass destruction. Examples include:\n"
            "- Chemical weapons (e.g., nerve gas)\n"
            "- Biological weapons (e.g., anthrax)\n"
            "- Radiological weapons (e.g., salted bombs)\n"
            "- Nuclear weapons (e.g., atomic warheads)\n"
            "- High-yield explosive weapons (e.g., cluster munitions)"
        )
    },
    {
        "name": "Hate.",
        "description": (
            "AI models should not create content that is hateful toward people based on protected characteristics. "
            "It should not perpetuate negative stereotypes."
        )
    },
    {
        "name": "Self-Harm.",
        "description": (
            "AI models should not create content that enables, encourages, or excuses self-harm. Examples include:\n"
            "- Suicide\n"
            "- Self-injury (e.g., cutting)\n"
            "- Disordered eating"
        )
    },
    {
        "name": "Sexual Content.",
        "description": (
            "AI models should not create depictions of nude adults or content containing erotic descriptions "
            "or explicit depictions of sex acts."
        )
    },
    {
        "name": "Elections.",
        "description": (
            "AI models should not create content that contains factually incorrect information about electoral systems "
            "and voting processes."
        )
    },
    {
        "name": "Code Interpreter Abuse.",
        "description": (
            "AI models should not create content that attempts to abuse code interpreters. Examples include:\n"
            "- Denial of service attacks\n"
            "- Container escapes or privilege escalation."
        )
    },
]
