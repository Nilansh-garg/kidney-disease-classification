// =============================================
//   KidneyHealth AI — contact.js
//   Contact Form Validation & Submission
// =============================================

const contactForm = document.getElementById('contactForm');

if (contactForm) {
    contactForm.addEventListener('submit', handleFormSubmit);
}

// --- Form Submit Handler ---
function handleFormSubmit(e) {
    e.preventDefault();

    const formData = {
        name:    document.getElementById('name')?.value.trim()    || '',
        email:   document.getElementById('email')?.value.trim()   || '',
        subject: document.getElementById('subject')?.value.trim() || '',
        message: document.getElementById('message')?.value.trim() || ''
    };

    const errors = validateForm(formData);

    if (errors.length > 0) {
        showFormErrors(errors);
        return;
    }

    const submitBtn = contactForm.querySelector('button[type="submit"]');
    const origText  = submitBtn.innerHTML;
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';

    // Simulate API call (replace with real endpoint if available)
    setTimeout(() => {
        showSuccessMessage();
        contactForm.reset();
        clearAllInputErrors();
        clearCharCounter();

        submitBtn.disabled = false;
        submitBtn.innerHTML = origText;

        console.log('Contact form submitted:', formData);
    }, 1600);
}

// --- Validation ---
function validateForm(data) {
    const errors = [];

    if (data.name.length < 2)
        errors.push('Full name must be at least 2 characters long');

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(data.email))
        errors.push('Please enter a valid email address');

    if (data.subject.length < 5)
        errors.push('Subject must be at least 5 characters long');

    if (data.message.length < 20)
        errors.push('Message must be at least 20 characters long');

    return errors;
}

function showFormErrors(errors) {
    removeExistingMsg('.form-errors');

    const html = `
        <div class="form-errors" style="
            background:#fef2f2;border:1px solid #ef444455;border-left:4px solid #ef4444;
            border-radius:10px;padding:1rem 1.25rem;margin-bottom:1.25rem;color:#b91c1c;">
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;font-weight:700;">
                <i class="fas fa-exclamation-circle"></i> Please fix the following:
            </div>
            <ul style="margin:0;padding-left:1.5rem;">
                ${errors.map(e => `<li style="margin-bottom:0.25rem;font-size:0.9rem;">${e}</li>`).join('')}
            </ul>
        </div>
    `;

    contactForm.insertAdjacentHTML('afterbegin', html);
    contactForm.querySelector('.form-errors')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function showSuccessMessage() {
    removeExistingMsg('.form-errors');
    removeExistingMsg('.form-success');

    const html = `
        <div class="form-success" style="
            background:#f0fdf4;border:1px solid #22c55e55;border-left:4px solid #22c55e;
            border-radius:10px;padding:1.5rem;margin-bottom:1.25rem;color:#15803d;
            text-align:center;animation:formFadeIn 0.4s ease;">
            <i class="fas fa-check-circle" style="font-size:2.5rem;color:#22c55e;margin-bottom:0.75rem;display:block;"></i>
            <strong style="display:block;margin-bottom:0.25rem;font-size:1.05rem;">Message Sent!</strong>
            <span style="font-size:0.9rem;color:#16a34a;">Thank you for reaching out. We'll get back to you shortly.</span>
        </div>
    `;

    contactForm.insertAdjacentHTML('afterbegin', html);
    contactForm.querySelector('.form-success')?.scrollIntoView({ behavior: 'smooth', block: 'center' });

    setTimeout(() => {
        const msg = contactForm.querySelector('.form-success');
        if (msg) {
            msg.style.animation = 'formFadeOut 0.4s ease forwards';
            setTimeout(() => msg.remove(), 400);
        }
    }, 5500);
}

function removeExistingMsg(selector) {
    contactForm.querySelector(selector)?.remove();
}

// --- Real-time Validation ---
const inputIds = ['name', 'email', 'subject', 'message'];

inputIds.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;

    el.addEventListener('blur', () => validateInput(el));

    el.addEventListener('input', () => {
        el.style.borderColor = '';
        el.parentElement.querySelector('.input-error')?.remove();
    });
});

function validateInput(input) {
    const value = input.value.trim();
    let isValid = true;
    let errorMsg = '';

    switch (input.id) {
        case 'name':
            if (value.length < 2) { isValid = false; errorMsg = 'Name must be at least 2 characters'; }
            break;
        case 'email':
            if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) { isValid = false; errorMsg = 'Enter a valid email address'; }
            break;
        case 'subject':
            if (value.length < 5) { isValid = false; errorMsg = 'Subject must be at least 5 characters'; }
            break;
        case 'message':
            if (value.length < 20) { isValid = false; errorMsg = `${Math.max(0, 20 - value.length)} more characters needed`; }
            break;
    }

    if (!isValid) showInputError(input, errorMsg);
    return isValid;
}

function showInputError(input, message) {
    input.parentElement.querySelector('.input-error')?.remove();
    input.style.borderColor = '#ef4444';

    const div = document.createElement('div');
    div.className = 'input-error';
    div.style.cssText = 'color:#ef4444;font-size:0.82rem;margin-top:0.3rem;display:flex;align-items:center;gap:4px;';
    div.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${message}`;

    input.parentElement.appendChild(div);
}

function clearAllInputErrors() {
    inputIds.forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        el.style.borderColor = '';
        el.parentElement.querySelector('.input-error')?.remove();
    });
}

// --- Character Counter for Message ---
const messageField = document.getElementById('message');
let charCounter    = null;

if (messageField) {
    charCounter = document.createElement('div');
    charCounter.className = 'char-counter';
    charCounter.style.cssText = 'text-align:right;font-size:0.8rem;color:var(--text-light);margin-top:0.25rem;';
    messageField.parentElement.appendChild(charCounter);

    messageField.addEventListener('input', function() {
        const len = this.value.length;
        if (len < 20) {
            charCounter.textContent = `${20 - len} more characters needed`;
            charCounter.style.color = '#ef4444';
        } else {
            charCounter.textContent = `${len} characters`;
            charCounter.style.color = '#22c55e';
        }
    });
}

function clearCharCounter() {
    if (charCounter) {
        charCounter.textContent = '';
    }
}

// --- Animation Styles ---
const style = document.createElement('style');
style.textContent = `
    @keyframes formFadeIn {
        from { opacity: 0; transform: translateY(-8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes formFadeOut {
        from { opacity: 1; transform: translateY(0); }
        to   { opacity: 0; transform: translateY(-8px); }
    }
    .form-group input:focus,
    .form-group textarea:focus {
        outline: none;
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(13,110,110,0.12);
    }
`;
document.head.appendChild(style);

console.log('KidneyHealth AI — contact.js loaded');
