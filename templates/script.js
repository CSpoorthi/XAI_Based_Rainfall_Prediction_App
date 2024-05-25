document.addEventListener("DOMContentLoaded", function() {
    const rainAnimation = document.querySelector('.rain-animation');

    // Create a large number of raindrops and add them to the rain-animation div
    for (let i = 0; i < 150; i++) {
        const raindrop = document.createElement('div');
        raindrop.classList.add('raindrop');
        raindrop.style.left = `${Math.random() * 100}%`;
        raindrop.style.animationDuration = `${Math.random() * 1 + 0.5}s`; // Vary animation duration for realistic effect
        rainAnimation.appendChild(raindrop);
    }
});

