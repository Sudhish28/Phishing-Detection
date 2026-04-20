import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import re
import os
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


from wordcloud import WordCloud

print("=" * 60)
print("  PHISHING EMAIL DETECTION - NLP & MACHINE LEARNING")
print("=" * 60)
print("\n[STEP 1] Generating dataset...")

np.random.seed(42)


phishing_subjects = [
    "URGENT: Your account has been compromised",
    "Action required: Verify your identity now",
    "Your PayPal account is suspended",
    "Congratulations! You have won $1,000,000",
    "Important: Update your banking information",
    "Security Alert: Unusual login detected",
    "Final notice: Your account will be deleted",
    "Claim your prize before it expires",
    "Your package could not be delivered - click here",
    "Immediate action needed: Password expiring",
    "IRS Tax Refund Notification",
    "Netflix: Your payment was declined",
    "Verify your email or lose access forever",
    "You have been selected for a special reward",
    "Click here to confirm your identity",
]

phishing_bodies = [
    "Dear Customer, We have detected suspicious activity on your account. Click the link below IMMEDIATELY to verify your identity or your account will be permanently suspended. http://secure-login-verify.xyz/confirm",
    "CONGRATULATIONS! You have been selected as a winner. To claim your $500 gift card, provide your personal details and credit card number at: http://freegift-claim.net/winner",
    "Your PayPal account has been limited due to unusual activity. Please confirm your information within 24 hours to avoid suspension: http://paypal-secure-verify.ru/login",
    "URGENT NOTICE from IRS: You owe back taxes. Failure to respond in 48 hours will result in arrest. Call 1-800-FAKE-IRS or click here: http://irs-refund-claim.xyz",
    "Dear Valued User, Your bank account requires immediate verification. Provide SSN, account number, and password to restore access: http://bankofamerica-login.cn/verify",
    "Alert! Someone attempted to login to your account from Russia. Verify now or we will disable your account: http://account-security-alert.net/verify",
    "You have a pending package from FedEx. Pay the $2.99 delivery fee now to release your package: http://fedex-delivery-confirm.xyz/pay",
    "We noticed your Netflix payment failed. Update your billing info to continue your subscription: http://netflix-billing-update.net/pay",
    "Your Microsoft account password expires in 24 hours. Click to reset now: http://microsoft-password-reset.xyz/update",
    "Congratulations! Apple has selected you for an exclusive offer. Claim your free iPhone 15 at: http://apple-reward-giveaway.net/claim",
    "Your email storage is 99% full. Upgrade NOW or all your emails will be deleted: http://email-storage-upgrade.xyz/buy",
    "FINAL WARNING: Your social security number has been suspended. Call immediately to avoid criminal charges: http://ssa-verify.cn/case",
    "Dear customer, your Amazon account was accessed from an unknown device. Secure your account: http://amazon-security-alert.xyz/lock",
    "You have won a lottery prize of $10,000! To collect, provide your bank routing number at: http://lottery-winner-claim.net/collect",
    "IMPORTANT: Your Google account will be terminated unless you verify your identity in 12 hours: http://google-verify-now.xyz/confirm",
]

legit_subjects = [
    "Team meeting scheduled for Friday 3pm",
    "Project update: Q3 milestones achieved",
    "Welcome to our newsletter!",
    "Your order has shipped - tracking #12345",
    "Meeting recap and action items",
    "Invitation: Company picnic next Saturday",
    "Monthly report attached",
    "Re: Feedback on the draft document",
    "Python conference registration confirmed",
    "Your subscription renewal is due next month",
    "Happy birthday from the team!",
    "Job application received - thank you",
    "Your appointment is confirmed for Monday",
    "Reminder: Submit timesheets by Friday",
    "New blog post: 5 tips for better coding",
]

legit_bodies = [
    "Hi team, Just a reminder that our weekly sync is scheduled for Friday at 3:00 PM in Conference Room B. Please bring your status updates. Let me know if you have any conflicts.",
    "Hello everyone, I am happy to share that we have successfully completed all Q3 milestones ahead of schedule. Great work by everyone on the team. More details in the attached report.",
    "Welcome to our monthly newsletter! This month we cover the latest trends in cybersecurity, upcoming webinars, and tips for staying safe online. Unsubscribe at any time.",
    "Your Amazon order #112-3456789 has shipped! Your package is expected to arrive by Thursday. You can track your shipment using the tracking number above.",
    "Hi, following up from our meeting yesterday. Action items: 1) Alice to finalize the design by Wednesday. 2) Bob to review the budget. 3) Carol to schedule client call.",
    "Hi all, We are hosting our annual company picnic on Saturday June 15 at Riverside Park from noon to 4 PM. Food and drinks will be provided. RSVPs due by June 10.",
    "Please find attached the monthly analytics report for May. Highlights include a 12% increase in user engagement and a 5% reduction in support tickets compared to April.",
    "Hi Sarah, Thanks for sending over the draft. I reviewed it and have a few suggestions. Overall the structure is solid. I left comments in the document for your reference.",
    "Your registration for PyCon 2024 is confirmed! The conference runs from April 19-21 in Pittsburgh. Your badge and schedule will be emailed two weeks before the event.",
    "This is a friendly reminder that your annual subscription renews on July 1st. No action is needed if you wish to continue. Contact us at support@company.com to cancel.",
    "Dear John, Wishing you a very happy birthday from the entire team! We hope you enjoy the small celebration we have planned for the lunch break today. See you soon!",
    "Dear Applicant, Thank you for applying to the Software Engineer position at TechCorp. We have received your application and will review it within two weeks.",
    "This is a confirmation for your dentist appointment on Monday, June 3rd at 10:00 AM with Dr. Smith. Please arrive 10 minutes early to fill out paperwork.",
    "Hi team, just a reminder to submit your timesheets before end of day Friday. Please ensure all project codes are correctly entered. Contact HR if you have questions.",
    "New post on our blog: 5 Practical Tips for Writing Cleaner Python Code. Topics include PEP 8 guidelines, docstrings, list comprehensions, and more. Read now on our website.",
]

phishing_emails = []
for i in range(500):
    subj = phishing_subjects[i % len(phishing_subjects)]
    body = phishing_bodies[i % len(phishing_bodies)]
    # Add slight variation
    variation = f" Reference ID: {np.random.randint(10000,99999)}." if i % 3 == 0 else ""
    phishing_emails.append(subj + " " + body + variation)

legit_emails = []
for i in range(500):
    subj = legit_subjects[i % len(legit_subjects)]
    body = legit_bodies[i % len(legit_bodies)]
    variation = f" Best regards, Team {np.random.randint(1,20)}." if i % 3 == 0 else ""
    legit_emails.append(subj + " " + body + variation)

emails = phishing_emails + legit_emails
labels = [1] * 500 + [0] * 500   # 1 = Phishing, 0 = Legitimate

df = pd.DataFrame({'email_text': emails, 'label': labels})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  

print(f"  ✓ Dataset created: {len(df)} emails")
print(f"  ✓ Phishing emails : {df['label'].sum()} ({df['label'].mean()*100:.0f}%)")
print(f"  ✓ Legit emails    : {(df['label']==0).sum()} ({(df['label']==0).mean()*100:.0f}%)")


df.to_csv('phishing_dataset.csv', index=False)
print("  ✓ Dataset saved to phishing_dataset.csv")



print("\n[STEP 2] Preprocessing text...")

def preprocess_text(text):
    """
    Clean and normalize email text:
    1. Lowercase all words
    2. Remove URLs (http links)
    3. Remove special characters and punctuation
    4. Remove numbers
    5. Remove extra whitespace
    6. Remove stopwords
    """
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', ' url_link ', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' email_address ', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords (using sklearn's English stopwords)
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    return ' '.join(words)

df['cleaned_text'] = df['email_text'].apply(preprocess_text)
print(f"  ✓ Text cleaning complete")
print(f"  ✓ Sample cleaned text (first 100 chars):")
print(f"    '{df['cleaned_text'].iloc[0][:100]}...'")



print("\n[STEP 3] Extracting TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=5000,     # Top 5000 most important words
    ngram_range=(1, 2),    # Unigrams and bigrams
    min_df=2,              # Word must appear in at least 2 documents
    max_df=0.95,           # Ignore words in 95%+ of documents
    sublinear_tf=True      # Apply log normalization
)

X = tfidf.fit_transform(df['cleaned_text'])
y = df['label']

print(f"  ✓ TF-IDF matrix shape: {X.shape}")
print(f"  ✓ Vocabulary size: {len(tfidf.vocabulary_)} unique terms")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  ✓ Training set: {X_train.shape[0]} samples")
print(f"  ✓ Testing  set: {X_test.shape[0]} samples")


# ─────────────────────────────────────────
# STEP 4 – MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────
print("\n[STEP 4] Training and evaluating models...")

models = {
    'Naive Bayes': MultinomialNB(alpha=0.1),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

results = {}

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    # Metrics
    acc   = accuracy_score(y_test, y_pred)
    auc   = roc_auc_score(y_test, y_prob)
    cv    = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    cm    = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Legitimate','Phishing'])

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': acc,
        'auc': auc,
        'cv_score': cv,
        'confusion_matrix': cm,
        'report': report
    }

    print(f"\n  --- {name} ---")
    print(f"  Accuracy   : {acc*100:.2f}%")
    print(f"  ROC-AUC    : {auc:.4f}")
    print(f"  CV Score   : {cv*100:.2f}%")
    print(f"\n{report}")


# ─────────────────────────────────────────
# STEP 5 – VISUALIZATIONS
# ─────────────────────────────────────────
print("\n[STEP 5] Generating visualizations...")
os.makedirs('figures', exist_ok=True)

COLORS = {
    'phishing': '#E74C3C',
    'legit':    '#2ECC71',
    'blue':     '#3498DB',
    'dark':     '#2C3E50',
    'light':    '#ECF0F1',
    'orange':   '#E67E22',
    'purple':   '#9B59B6',
}

# --- FIG 1: Dataset Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Figure 1: Dataset Overview', fontsize=16, fontweight='bold', color=COLORS['dark'])

# Pie chart
labels_pie = ['Legitimate (50%)', 'Phishing (50%)']
colors_pie = [COLORS['legit'], COLORS['phishing']]
axes[0].pie([500, 500], labels=labels_pie, colors=colors_pie,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[0].set_title('Email Class Distribution', fontsize=13, color=COLORS['dark'])

# Email length distribution
df['text_length'] = df['email_text'].apply(len)
for label_val, color, name in [(0, COLORS['legit'], 'Legitimate'), (1, COLORS['phishing'], 'Phishing')]:
    subset = df[df['label'] == label_val]['text_length']
    axes[1].hist(subset, bins=30, alpha=0.6, color=color, label=name, edgecolor='white')
axes[1].set_xlabel('Email Text Length (characters)', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('Email Length Distribution', fontsize=13, color=COLORS['dark'])
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig1_dataset_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Figure 1 saved")

# --- FIG 2: Word Clouds ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Figure 2: Most Frequent Words', fontsize=16, fontweight='bold', color=COLORS['dark'])

for idx, (label_val, title, color) in enumerate([
    (1, 'Phishing Emails', COLORS['phishing']),
    (0, 'Legitimate Emails', COLORS['legit'])
]):
    text_all = ' '.join(df[df['label'] == label_val]['cleaned_text'])
    wc = WordCloud(width=600, height=400, background_color='white',
                   colormap='Reds' if label_val == 1 else 'Greens',
                   max_words=80, collocations=False).generate(text_all)
    axes[idx].imshow(wc, interpolation='bilinear')
    axes[idx].axis('off')
    axes[idx].set_title(title, fontsize=14, color=color, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('figures/fig2_word_clouds.png', dpi=150, bbox_inches='tight')

for idx, (name, res) in enumerate(results.items()):
    cm = res['confusion_matrix']
    im = axes[idx].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[idx].set_title(f'{name}\nAccuracy: {res["accuracy"]*100:.1f}%',
                        fontsize=12, color=COLORS['dark'])
    tick_marks = [0, 1]
    axes[idx].set_xticks(tick_marks)
    axes[idx].set_yticks(tick_marks)
    axes[idx].set_xticklabels(['Legit', 'Phishing'], fontsize=10)
    axes[idx].set_yticklabels(['Legit', 'Phishing'], fontsize=10)
    axes[idx].set_ylabel('True Label', fontsize=10)
    axes[idx].set_xlabel('Predicted Label', fontsize=10)
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            axes[idx].text(j, i, format(cm[i, j], 'd'),
                           ha='center', va='center', fontsize=14,
                           color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.savefig('figures/fig3_confusion_matrices.png', dpi=150, bbox_inches='tight')
fig, ax = plt.subplots(figsize=(8, 6))
colors_roc = [COLORS['blue'], COLORS['orange'], COLORS['purple']]

for (name, res), color in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, color=color, lw=2.5,
            label=f'{name} (AUC = {res["auc"]:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Figure 4: ROC Curves - All Models', fontsize=14, fontweight='bold', color=COLORS['dark'])
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])

plt.tight_layout()
plt.savefig('figures/fig4_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Figure 4 saved")

# --- FIG 5: Model Comparison Bar Chart ---
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Figure 5: Model Performance Comparison', fontsize=16, fontweight='bold', color=COLORS['dark'])

model_names = list(results.keys())
metrics = {
    'Accuracy (%)':   [r['accuracy'] * 100 for r in results.values()],
    'ROC-AUC':        [r['auc'] for r in results.values()],
    'CV Score (%)':   [r['cv_score'] * 100 for r in results.values()],
}

bar_colors = [COLORS['blue'], COLORS['orange'], COLORS['purple']]

for ax, (metric_name, values) in zip(axes, metrics.items()):
    bars = ax.bar(model_names, values, color=bar_colors, edgecolor='white',
                  width=0.5, linewidth=1.5)
    ax.set_title(metric_name, fontsize=13, color=COLORS['dark'])
    ax.set_ylim(0, 110 if '%' in metric_name else 1.1)
    ax.set_ylabel(metric_name, fontsize=10)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)
    # Value labels on bars
    for bar, val in zip(bars, values):
        label = f'{val:.1f}%' if '%' in metric_name else f'{val:.3f}'
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/fig5_model_comparison.png', dpi=150, bbox_inches='tight')

feature_names = np.array(tfidf.get_feature_names_out())
# Use Logistic Regression coefficients
lr_model = results['Logistic Regression']['model']
coef = lr_model.coef_[0]

top_phishing_idx = np.argsort(coef)[-15:][::-1]
top_legit_idx    = np.argsort(coef)[:15]

for ax, idx_list, color, title in [
    (axes[0], top_phishing_idx, COLORS['phishing'], 'Top Phishing Indicators'),
    (axes[1], top_legit_idx,    COLORS['legit'],    'Top Legitimate Indicators'),
]:
    words = feature_names[idx_list]
    scores = np.abs(coef[idx_list])
    bars = ax.barh(range(len(words)), scores, color=color, edgecolor='white', alpha=0.85)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=10)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=13, color=COLORS['dark'], fontweight='bold')
    ax.set_xlabel('TF-IDF Coefficient Weight', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig6_top_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Figure 6 saved")


# ─────────────────────────────────────────
# STEP 6 – LIVE PREDICTION DEMO
# ─────────────────────────────────────────
print("\n[STEP 6] Running prediction demo...")

best_model = results['Logistic Regression']['model']

test_emails = [
    "URGENT: Your bank account has been suspended. Click here immediately to verify your identity: http://secure-bank-verify.xyz/login",
    "Hi team, the project meeting is scheduled for Thursday at 2 PM in the conference room. Please bring your weekly status reports.",
    "Congratulations! You have won a $500 Amazon gift card. Claim it now at http://amazon-giftcard-free.net before it expires!",
    "Dear John, your subscription renewal is coming up next month. No action needed unless you wish to make changes.",
]

print("\n  Live Prediction Results:")
print("  " + "=" * 55)
for email in test_emails:
    cleaned = preprocess_text(email)
    vec = tfidf.transform([cleaned])
    prediction = best_model.predict(vec)[0]
    confidence = best_model.predict_proba(vec)[0].max() * 100
    label_str = "⚠  PHISHING" if prediction == 1 else "✓ LEGITIMATE"
    print(f"  Email: '{email[:60]}...'")
    print(f"  → {label_str}  (Confidence: {confidence:.1f}%)")
    print()

print("\n" + "=" * 60)
print("  PROJECT COMPLETE — ALL OUTPUTS SAVED TO figures/")
print("=" * 60)
