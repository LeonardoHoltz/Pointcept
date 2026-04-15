
export function showToast(message, duration = 3000) {
  const container = document.getElementById("toast-container");

  const toast = document.createElement("div");
  toast.className = "toast";
  toast.innerText = message;

  container.appendChild(toast);
  toast.offsetHeight;

  // animação de entrada
  requestAnimationFrame(() => {
    toast.classList.add("show");
  });

  // remover depois
  setTimeout(() => {
    toast.classList.remove("show");
    toast.addEventListener("transitionend", () => toast.remove());
  }, duration);
}