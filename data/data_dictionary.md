# Data Dictionary

## glass_reviews.csv
Raw employee reviews and workplace ratings for Amazon Fulfillment Centers collected from Glassdoor. Contains quantitative ratings, employee employment characteristics, and qualitative feedback (pros, cons, advice to management).

| Column | Type | Description |
| --- | --- | --- |
| Unfair Workplace Practices and Lack of Growth Opportunities and Poor Work Conditions and Management Support: | int | Unique review record identifier (indexed 1 to 127) |
| rating_date | string | Timestamp when the review was posted on Glassdoor (ISO 8601 format) |
| count_helpful | int | Number of readers who marked the review as helpful |
| count_unhelpful | int | Number of readers who marked the review as unhelpful |
| employee_length | int | Duration of employment at Amazon (in years) |
| employee_status | string | Employment status category (e.g., REGULAR, PART_TIME) |
| employee_type | string | Employment relationship status (e.g., Current employee, Former employee) |
| flag_covid | boolean | Indicator if the review references COVID-19 pandemic impacts |
| flag_featured | boolean | Indicator if the review is featured or highlighted on Glassdoor |
| flags_business_outlook | string | Reviewer outlook on company business future (POSITIVE, NEUTRAL, NEGATIVE) |
| flags_ceo_approval | string | Reviewer opinion on CEO leadership (APPROVE, DISAPPROVE, NO_OPINION) |
| flags_recommend_frend | string | Recommendation to a friend regarding working at the company (POSITIVE, NEGATIVE) |
| rating_culture_values | int | Sub-rating for company culture and values (scale 0-5; 0 indicates unrated) |
| rating_diversity_inclusion | int | Sub-rating for diversity and inclusion (scale 0-5; 0 indicates unrated) |
| rating_overall | int | Overall star rating given by employee to Amazon Fulfillment Center (scale 1-5) |
| rating_work_life | int | Sub-rating for work-life balance (scale 0-5; 0 indicates unrated) |
| summary | string | Headline or summary title written by reviewer |
| company_name | string | Name of employer entity evaluated (Amazon Fulfillment Center) |
| review_advice | float | Additional advice field (unpopulated in raw extraction) |
| career_opportunities_rating | float | Alternate rating field for career growth opportunities (unpopulated) |
| employee_location | string | Job location (city and state) of fulfillment center |
| employee_job_title | string | Job role or title of reviewer (e.g., Warehouse Fulfillment Associate, Packer, Stower) |
| advice_to_management | string | Free-form feedback and direct advice to Amazon management |
| review_pros | string | Free-form comments detailing positive aspects of working at Amazon |
| review_cons | string | Free-form comments detailing challenges, pain points, and negative experiences |
| rating_compensation_benefits | int | Sub-rating for compensation and benefits (scale 0-5; 0 indicates unrated) |
| rating_senior_leadership | int | Sub-rating for senior leadership and management effectiveness (scale 0-5; 0 indicates unrated) |
| rating_career_opportunities | int | Sub-rating for career growth and advancement opportunities (scale 0-5; 0 indicates unrated) |

---

## youtube_transcripts.csv
Raw transcripts and video title metadata extracted from YouTube videos discussing Amazon warehouse working conditions, employee roles, and operational experiences.

| Column | Type | Description |
| --- | --- | --- |
| Folder name | string | Title or topic identifier of the YouTube video |
| original_script | string | Complete raw, verbatim spoken transcript text extracted from the video |
