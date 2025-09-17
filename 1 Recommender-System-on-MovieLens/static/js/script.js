// Enhanced search functionality
document.addEventListener('DOMContentLoaded', function() {
    const movieInput = document.getElementById('movie_title');
    const form = document.querySelector('.search-form');
    const submitBtn = document.querySelector('.submit-btn');

    // Add loading state to form submission
    if (form) {
        form.addEventListener('submit', function() {
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Searching...';
            submitBtn.disabled = true;
        });
    }

    // Auto-suggest functionality (if you want to implement live search)
    if (movieInput) {
        let searchTimeout;
        
        movieInput.addEventListener('input', function() {
            clearTimeout(searchTimeout);
            const query = this.value.trim();
            
            if (query.length >= 2) {
                searchTimeout = setTimeout(() => {
                    // Implement live search if needed
                    console.log('Searching for:', query);
                }, 300);
            }
        });
    }

    // Animate movie cards on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe movie cards
    document.querySelectorAll('.movie-card').forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(card);
    });

    // Add click analytics (optional)
    document.querySelectorAll('.movie-card').forEach(card => {
        card.addEventListener('click', function() {
            const movieTitle = this.querySelector('.movie-title').textContent;
            console.log('Movie clicked:', movieTitle);
        });
    });
});
