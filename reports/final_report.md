## Statistical Validation (A/B Testing)
To optimize our pricing, we tested several traditional risk segments. Surprisingly, our statistical tests (T-tests and Chi-Squared) showed that **Gender, Province, and ZipCode are not primary drivers of risk** in our current portfolio (all p-values > 0.05).

### Business Impact:
- **Pricing Strategy:** I recommend against implementing aggressive "regional" or "gender-based" discounts at this time, as the evidence does not support a significant difference in claim frequency or severity.
- **Next Steps:** Because simple segments didn't work, we are moving toward **Machine Learning models** to uncover complex, multi-variable risk patterns that simple A/B tests might miss.