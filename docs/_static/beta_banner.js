/* Inject a sitewide beta banner at the top of every page */
document.addEventListener("DOMContentLoaded", function () {
    var banner = document.createElement("div");
    banner.className = "beta-banner";
    banner.innerHTML =
        '<span class="beta-banner__icon">\uD83D\uDEA7</span>' +
        '<div class="beta-banner__text">' +
        '<strong>Beta Documentation</strong> &mdash; ' +
        "This is the beta version of the CUSTOMHyS documentation, " +
        "automatically generated from the source code. " +
        "We are actively working on expanding and improving it. " +
        "Content may be incomplete or subject to change." +
        "</div>";

    /* RTD theme: article body lives here */
    var target =
        document.querySelector('div[itemprop="articleBody"]') ||
        document.querySelector(".rst-content .document") ||
        document.querySelector(".document");

    if (target) {
        target.insertBefore(banner, target.firstChild);
    }
});
