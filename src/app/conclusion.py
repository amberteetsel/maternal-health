import os
import streamlit as st

# Display Function
def render_conclusion(conclusion_pipeline: list):

    for x in conclusion_pipeline:

        st.subheader(x.get('head'))
        st.markdown(x.get('body'))

# Inputs
p1_head = "Structural Polarization and Predictive Legal Landscapes"
p1_body = """
The data reveals a profound, undeniable truth: a state's legal and political stance on reproductive rights is directly stamped onto
the physical health of its citizens. By looking strictly at anonymous health outcomes, such as inadequate prenatal care, maternity care
deserts, and infant mortality, it is possible to predict with near-perfect accuracy (up to 98%) whether a state restricts or protects abortion access.
This means that states restricting reproductive healthcare do not merely have slightly worse metrics; they exist in an entirely separate,
degraded clinical reality. In these states, families face a structurally hostile environment characterized by higher baseline mortality,
fewer active OB/GYNs, and widespread care deserts.
"""

p2_head = "Shifting Legal Landscapes"
p2_body = """
    The current state of polarization is not a sudden anomaly. It resulted from a decades-long chain reaction. When analyzing nearly
    50 years of national pregnany, birth, abortion, and miscarriage rates, the data naturally groups itself into four distinct, 
    back-to-back historical eras. Without ever knowing the calendar years, the data mapped out clear boundaries aligned with major
    political shifts. The first era, Post-*Roe*, spans 1973-1976 and represents a period of rapid adaption to newly legalized care.
    The next era (Stablization) represents a long-term leveling out of public health care trends from 1977 to 1996. Next, the popularization
    of long-acting, reliable contraceptives from 1997-2010 created a period where teen pregnancies and unintended pregnancies decreased.
    Finally, the period from 2011-2020 saw a wave of aggressive Targeted Regulation of Abortion Providers (TRAP) laws that greatly
    restricted abortion access and set the stage for the 2022 *Dobbs* decision. Court decisions and state laws are not abstract legal
    debates: they are the primary forces shaping the physical well-being of women and families across generations.
"""

p3_head = "Danger in the South"
p3_body = """
    When examining healthcare policies and outcomes on a map, geography alone becomes a powerful predictor of maternal emergencies.
    The United States is divided into four census regions that closely align to distinct health ecosystems. 
    The West and Northeast present as highly protected, low-risk regions. In contrast, the South and parts of the Midwest emerge as high-risk
    zones with severely limited access to care. The real-worl danger of this divide is stark: simply seeking emergency care in the restricted
    South increases a patient's risk of experiencing a severe, life-threatening complication—such as sepsis, eclampsia, or heavy
    hemorrhaging—by roughly 33% to 35% compared to the Northeast, regardless of their age or race.
"""

p4_head = "Risk Factors Do Not Exist in a Vacuum"
p4_body = """
    At the individual patient level, analysis of intensive care unit (ICU) admissions shows that maternal emergencies are rarely
    caused by a single, isolated issue. Health risks compound and feed into one another. A patient experiencing preeclampsia, gestational
    diabetes, and the challenges of older maternal age faces an escalating, interconnected spiral of danger. Standard healthcare assessments
    often fail when they treat these risks as completely separate. To save lives, both medical practices and political policies must
    recognize that clinical vulnerabilities are deeply layered and connected.
"""

p5_head = "The Human Cost"
p5_body = """
    Ultimately, the numbers bridge the gap between cold statistics and the tragic human realities unfolding across the country.
    When vague criminal penalties collide with medical decision-making, doctors are forced to hesitate, turning standard, treatable
    complications into fatal emergencies. These systemic failures do not affect everyone equally; historical disparities mean that Black
    and Native mothers face the highest baseline risks, which are then drastically multiplied under restrictive laws.
    Whether restrictive laws directly dismantle clinic networks or simply signal a broader political abandonment of public health, the
    conclusion is absolute: a state's legal restrictions on reproductive rights are structurally inseparable from the rising rates of
    suffering and death among its mothers.
"""

conclusion_all = [
    {'head': p1_head, 'body': p1_body},
    {'head': p2_head, 'body': p2_body},
    {'head': p3_head, 'body': p3_body},
    {'head': p4_head, 'body': p4_body},
    {'head': p5_head, 'body': p5_body}
]