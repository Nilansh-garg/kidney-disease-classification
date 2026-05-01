// =============================================
//   KidneyHealth AI — main.js
//   Global JS: Navigation, Animations, Stats
// =============================================

// --- Mobile Navigation ---
const hamburger = document.querySelector('.hamburger');
const navMenu   = document.querySelector('.nav-menu');

if (hamburger) {
    hamburger.addEventListener('click', () => {
        navMenu.classList.toggle('active');
        hamburger.classList.toggle('active');
    });

    document.querySelectorAll('.nav-menu a').forEach(link => {
        link.addEventListener('click', () => {
            navMenu.classList.remove('active');
            hamburger.classList.remove('active');
        });
    });
}

// Close menu on outside click
document.addEventListener('click', (e) => {
    if (navMenu && hamburger) {
        if (!navMenu.contains(e.target) && !hamburger.contains(e.target)) {
            navMenu.classList.remove('active');
            hamburger.classList.remove('active');
        }
    }
});

// --- Navbar scroll shadow ---
window.addEventListener('scroll', () => {
    const navbar = document.querySelector('.navbar');
    if (!navbar) return;
    if (window.scrollY > 40) {
        navbar.style.boxShadow = '0 4px 24px rgba(13,110,110,0.15)';
    } else {
        navbar.style.boxShadow = 'none';
    }
}, { passive: true });

// --- Smooth Scrolling ---
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            e.preventDefault();
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});

// --- Animated Counter for Stats ---
function animateCounter(el, target, isDecimal) {
    const duration  = 2000;
    const steps     = 80;
    const stepTime  = duration / steps;
    const increment = target / steps;
    let current     = 0;

    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            current = target;
            clearInterval(timer);
        }
        el.textContent = isDecimal
            ? current.toFixed(1)
            : Math.floor(current).toLocaleString();
    }, stepTime);
}

// Intersection Observer for stats
const statsSection = document.querySelector('.stats');
if (statsSection) {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.querySelectorAll('.stat-number').forEach(el => {
                    const target    = parseFloat(el.getAttribute('data-target'));
                    const isDecimal = el.getAttribute('data-decimal') === 'true';
                    animateCounter(el, target, isDecimal);
                });
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.4 });

    observer.observe(statsSection);
}

// --- Scroll Reveal Animation ---
const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity  = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, { threshold: 0.15 });

const animatedEls = document.querySelectorAll(
    '.feature-card, .step, .tech-card, .benefit-card, .use-case-card, .team-card, .spec-item, .timeline-item'
);

animatedEls.forEach((el, i) => {
    el.style.opacity    = '0';
    el.style.transform  = 'translateY(28px)';
    el.style.transition = `opacity 0.55s ease ${i * 0.07}s, transform 0.55s ease ${i * 0.07}s`;
    revealObserver.observe(el);
});

// --- FAQ Accordion ---
document.querySelectorAll('.faq-question').forEach(question => {
    question.addEventListener('click', () => {
        const item   = question.parentElement;
        const answer = item.querySelector('.faq-answer');
        const isOpen = item.classList.contains('active');

        // Close all
        document.querySelectorAll('.faq-item').forEach(i => {
            i.classList.remove('active');
            i.querySelector('.faq-answer').style.display = 'none';
        });

        // Toggle clicked
        if (!isOpen) {
            item.classList.add('active');
            answer.style.display = 'block';
        }
    });
});

// --- Floating Cards Animation Delay ---
document.querySelectorAll('.floating-card').forEach((card, i) => {
    card.style.animationDelay = `${i * 0.5}s`;
});

// --- Lazy Loading Images ---
if ('IntersectionObserver' in window) {
    const imgObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                if (img.dataset.src) {
                    img.src = img.dataset.src;
                    img.classList.add('loaded');
                    imgObserver.unobserve(img);
                }
            }
        });
    });

    document.querySelectorAll('img[data-src]').forEach(img => {
        imgObserver.observe(img);
    });
}

// --- Debounce Utility ---
function debounce(fn, wait) {
    let timer;
    return function(...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), wait);
    };
}

window.addEventListener('resize', debounce(() => {
    // reserved for responsive logic
}, 250));

// --- Console Branding ---
console.log('%c KidneyHealth AI ', 'background:#0d6e6e;color:white;font-size:16px;padding:10px;border-radius:6px;');
console.log('%c Developed by Nilansh Garg · Supervised by Krish Naik ', 'background:#094f4f;color:rgba(255,255,255,0.8);font-size:11px;padding:5px;');
