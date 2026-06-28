# Text Sections for Website Introduction
import inspect

# Infographic - Commonwealth Fund, Maternal Mortality Comparison (international)
infogram_embed_html1 = """
<div class="infogram-embed" data-id="80a84092-43ff-46f5-a159-5ec782d8f07b" data-type="interactive" data-title="Insights into the U.S. Maternal Mortality Crisis: An International Comparison: Exhibit 1"></div>
<script>!function(e,n,i,s){var d="InfogramEmbeds";var o=e.getElementsByTagName(n)[0];if(window[d]&&window[d].initialized)window[d].process&&window[d].process();else if(!e.getElementById(i)){var r=e.createElement(n);r.async=1,r.id=i,r.src=s,o.parentNode.insertBefore(r,o)}}(document,"script","infogram-async","https://e.infogram.com/js/dist/embed-loader-min.js");</script>
"""
infogram_embed_html2 = """
<div class="infogram-embed" data-id="ef6762be-4964-4388-bac2-58cbb2d821ae" data-type="interactive" data-title="The U.S. Maternal Health Divide: The Limited Maternal Health Services and Worse Outcomes of States Proposing New Abortion Restrictions: Exhibit 4"></div>
<script>!function(e,n,i,s){var d="InfogramEmbeds";var o=e.getElementsByTagName(n)[0];if(window[d]&&window[d].initialized)window[d].process&&window[d].process();else if(!e.getElementById(i)){var r=e.createElement(n);r.async=1,r.id=i,r.src=s,o.parentNode.insertBefore(r,o)}}(document,"script","infogram-async","https://e.infogram.com/js/dist/embed-loader-min.js");</script>
"""

intro_p1 = inspect.cleandoc("""
    In June 2022, the Supreme Court of the United States issued its controversial ruling in 
    Dobbs v. Jackson Women's Health Organization, a historic decision that overturned nearly half a century of 
    federal precedent established by Roe v. Wade (1973).
    The Dobbs ruling eliminated the national right to abortion, effectively reverting authority to 
    regulate, restrict, and outlaw the practice back to individual states.
    As a result, the United States has become a fractured patchwork of reproductive healthcare access where 
    an individual's legal rights and medical choices are dependent on geographic location.
    According to policy tracking by the Guttmacher Institute (2024), more than a dozen states immediately 
    enacted total or near-total bans on abortion. 
    Conversely, many states moved to enshrine abortion protections in state law or state constitutions. This divergence
    in policy has created severe clinical confusion and legal gridlock, with healthcare providers left to navigate 
    vague and sometimes contradictory language regarding exceptions for the life or health of the mother.

    The Dobbs decision occurred during a worsening maternal health crisis in the United States, further exacerbating the issue.
    Data from the Centers for Disease Control and Prevention (CDC, 2023) indicate that the United States has the highest maternal 
    mortality rate among developed nations. There are also severe disparities affecting maternal mortality 
    for Black and Indigenous populations.
    Longitudinal tracking by the Guttmacher Institute demonstrates that unintended pregnancy rates fluctuate closely 
    with access to comprehensive family planning resources including contraceptive affordability and abortion services. 
    By exploring how changing state-level abortion restrictions intersect with existing maternal health infrasstructure 
    and maternal health outcomes, researchers can begin to quantify the real ramifications of restrictive abortion policies.
""")

intro_p2 = inspect.cleandoc("""
    Examining the direct correlation between restrictive healthcare policies and maternal health outcomes is critically 
    important because legislative interventions carry profound, life-altering consequences for pregnant individuals, 
    infants, and medical networks. Public health research consistently warns that the implementation of strict 
    abortion bans and narrow gestational limits can inadvertently increase maternal morbidity by delaying essential care 
    for obstetric complications, such as ectopic pregnancies, premature rupture of membranes, or incomplete miscarriages. 
    Furthermore, states enacting the most stringent restrictions often exhibit pre-existing systemic vulnerabilities, 
    such as high rates of uninsured residents, severe shortages of obstetricians and gynecologists, 
    and widespread "maternity care deserts." For example, the restriction of reproductive healthcare services 
    frequently leads to the closure of local clinics and rural labor units, compounding barriers to 
    routine prenatal care and leading to higher rates of low-birth-weight infants and preterm births. 
    The economic and psychological strains placed on individuals forced to carry unintended pregnancies to term, 
    or travel thousands of miles across state lines for care, introduce significant socioeconomic stressors that 
    undermine long-term household and community stability.<sup><b>[1]</b></sup>
    Additionally, the chilling effect on medical education and physician recruitment in states with
    severe criminal penalties for doctors threatens to destabilize the broader OB/GYN and pediatric healthcare
    workforce for decades to come. Quantitative data analysis provides an objective framework to move past polarized 
    political rhetoric and systematically evaluate the empirical impacts of these legal shifts on 
    tangible medical outcomes. By aligning longitudinal datasets tracking policy status—such as total bans,
    heartbeat bans, and protections—with public health records covering infant birth metrics and 
    emergency room utilization, this study establishes a clear, data-driven narrative.
""")

intro_p3 = inspect.cleandoc("""
    The modern reproductive healthcare landscape is the direct result of a judicial trilogy occuring over the
    previous half-century that mirrored broader socioeconomic and cultural transformations in American society.
    When the Supreme Court decided *Roe v. Wade* (1973), it was amid a national push from medical professionals
    to eliminate dangerous, "back-alley" abortions as well as second-wave feminists demanding more bodily
    autonomy. The *Roe* decision anchored the right to abortion in the Fourteenth Amendment's implied right to
    privacy and was celebrated as a landmark victory for gender equality and public health. However, critics 
    opposed it as unconstitutional and immoral. Two decades later, in an environment
    of increasing political polarization and attacks on abortion facilities, *Planned Parenthood of Southeastern
    Pennsylvania v. Casey* (1992) attempted a compromise: it upheld *Roe*'s central tenet of legal abortion while
    allowing states to implement restrictions such as waiting periods and parental consent. While viewed by
    supporters as a pragmatic compromise, critics on both the left and right argued it went too far or not far
    enough. Ultimately the socio-legal tension culminated in *Dobbs v. Jackson Women's Health Organization* (2022),
    a decision issued by an ultraconservative majority following years of incremental legislative erosion of
    abortion access. The *Dobbs* decision rejected both *Roe* and *Casey* in returning complete control of
    abortion regulation to individual states. Dissenting justices warned that stripping away an established
    constitutional right would inflict immediate, systemic harm on women's health and bodily sovereignty (Center
    for Reproductive Rights, 2022). This study attempts to quantify that harm.
""")

intro_p4 = """
    As noted, the drastic divergence in individual states' reproductive healthcare policies has led to clinical
    confusion and legal gridlock. Healthcare providers are under immense pressure to obey restrictive abortion policies,
    in some states facing legal or even criminal repercussions for even appearing to violate these laws. If there is
    an exception for the health or life of the mother, doctors are forced to decide when a mother is 'close enough' to
    death or irreversible health complications in order to perform a life-saving abortion. In practice, this quagmire
    has had devastating and sometimes fatal consequences for American women.

    In Georgia, 28-year old Amber Nicole Thurman developed acute sepsis after an incomplete medication abortion. However,
    the routine procedure to resolve this issue, called dilation and curettage (D&C), was made a felony with few
    exceptions. Amber's condition worsened in a hospital bed as doctors delayed the procedure by over 20 hours while 
    navigating the legal boundaries of Georgia's 6-week abortion ban. Amber died from preventable septic shock
    (ProPublica).

    Texas resident Josseli Barnica was pregnant when the state's six-week abortion ban went into effect. She suffered
    a miscarriage at 27 weeks, but was denied standard medical care for 40 hours, until the fetal heartbeat ceased. 
    Doctors cited abortion law as the reason for delayed care. Josseli died a preventable death from an infection three
    days later. Texas law threatens up to 99 years in prison for doctors who provide abortions (ProPublica).

    Amber and Josseli were both women of color, as were many women who suffered similar fates. Their stories underline
    that reproductive healthcare policy is not an abstract exercise: it has very real implications for all mothers.
"""

# data sources in more detail, hypotheses for the study
intro_p5 = """
    Anecdotal evidence is powerful, but insufficient to make a case to policymakers for a national law protecting the right to abortion.
    Rigorous quantification is required to prove Thurman's and Barnica's stories were not isolated incidents but rather the 
    inevitable result of current law. To that end, this study uses longitudinal health and policy-tracking data to evaluate the
    hypothesis that abortion bans and restrictions lead to worse outcomes for mothers. The Centers for Disease Control and Prevention (CDC)
    contributes multiple sources through their National Center for Health Statistics (NCHS), including emergency room records (filtered
    for pregnancy-related visits) and birth records. The emergency room data spans 2018 - 2022, providing a look at common issues and
    diagnoses for pregnant women of varying demographics (race, age, etc.) The birth records, spanning 2018 - 2024, are particularly useful;
    they report detailed maternal demographic information, track risk factors like eclampsia and hypertension, and record complications
    such as perineal lacerations, unplanned hysterectomies, and maternal morbidity. This data is supplemented by America's Health Rankings
    annual *Health of Women and Children Report, 2018 - 2025* which includes comprehensive state-level data on maternal health metrics like
    rates of unintended pregnancy, adequacy of maternal health infrastructure, maternal mortality. By examining health outcomes in relation
    to reproductive policy, this study expects to find a relationship between abortion bans and maternal morbidity. State-level policy
    data comes from Temple University's Center for Public Health Law Research and includes indicators for total bans and gestational limits
    as well as for explicit statutory or constitutional protections. Finally, the Guttmacher Institute has longitudinal data on national
    and state-level rates of pregnancy, birth, abortion, and miscarriage from 1973 to present day. Taken together, these sources provide
    a comprehensive look at reproductive health in the United States.
"""