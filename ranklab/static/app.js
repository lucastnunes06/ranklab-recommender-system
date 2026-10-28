"use strict";

// O painel consulta a API local. Não guarda preferências ou dados pessoais.
const byId = (id) => document.getElementById(id);
const percentage = (value) => `${Math.round(Number(value) * 100)}%`;
const currency = new Intl.NumberFormat("pt-BR", {style: "currency", currency: "BRL"});
let budget = 2;
let requestCounter = 0;

function updateBudgetButtons(selected) {
  budget = Number(selected.dataset.value);
  document.querySelectorAll("#budgets button").forEach((button) => {
    const active = button === selected;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}

function buildCard(restaurant, position) {
  const card = document.createElement("article");
  card.className = "restaurant";
  const index = document.createElement("span");
  index.className = "rankbadge";
  index.textContent = `#${position + 1}`;

  const info = document.createElement("div");
  info.className = "restaurant-info";
  const title = document.createElement("h3");
  title.textContent = restaurant.name;
  const details = document.createElement("div");
  details.className = "desc";
  details.textContent = `${restaurant.cuisine} · ${restaurant.eta_minutes} min · ★ ${restaurant.rating} · ${currency.format(restaurant.delivery_fee)} de entrega`;

  const score = document.createElement("span");
  score.className = "score";
  score.textContent = `Índice do modelo: ${restaurant.ml_probability.toFixed(3)} (não calibrado)`;
  const bar = document.createElement("div");
  bar.className = "bar";
  const fill = document.createElement("i");
  fill.style.width = `${Math.min(100, Math.max(0, restaurant.ml_probability * 100))}%`;
  bar.append(fill);

  const reasons = document.createElement("div");
  (restaurant.reason || []).forEach((reason) => {
    const chip = document.createElement("span");
    chip.className = "reason";
    chip.textContent = reason;
    reasons.append(chip);
  });

  info.append(title, details, score, bar, reasons);
  card.append(index, info);
  return card;
}

async function loadDiversityStudy() {
  const body = byId("studyRows");
  try {
    const response = await fetch("/api/diversity-study");
    if (!response.ok) throw new Error(`API retornou ${response.status}`);
    const study = await response.json();
    body.replaceChildren();
    for (const row of study.levels) {
      const tr = document.createElement("tr");
      const values = [
        percentage(row.diversity),
        row.ndcg_at_5.toFixed(4),
        row.unique_cuisines_top_5.toFixed(2),
        percentage(row.hit_rate_at_5),
      ];
      for (const value of values) {
        const td = document.createElement("td");
        td.textContent = value;
        tr.append(td);
      }
      body.append(tr);
    }
  } catch (error) {
    body.replaceChildren();
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 4;
    cell.textContent = `Estudo indisponível: ${error.message}`;
    row.append(cell);
    body.append(row);
  }
}


async function loadRecommendations() {
  // Identificador evita que uma resposta antiga sobrescreva filtros mais novos.
  const currentRequest = ++requestCounter;
  const params = new URLSearchParams({
    cuisine: byId("cuisine").value,
    budget: String(budget),
    rush: String(byId("rush").checked),
    vegetarian: String(byId("veg").checked),
    diversity: byId("diversity").value,
    max_eta: byId("eta").value,
  });
  const container = byId("cards");
  container.textContent = "Calculando recomendações...";
  byId("apply").disabled = true;
  try {
    const response = await fetch(`/api/recommendations?${params}`);
    if (!response.ok) throw new Error(`API retornou ${response.status}`);
    const payload = await response.json();
    if (currentRequest !== requestCounter) return;

    container.replaceChildren();
    payload.restaurants.forEach((restaurant, i) => {
      container.append(buildCard(restaurant, i));
    });
    if (!payload.restaurants.length) {
      container.textContent = payload.notice || "Nenhum restaurante atende aos filtros. Ajuste orçamento, dieta ou prazo.";
    }
    byId("count").textContent = `${payload.restaurants.length} sugestões`;
    byId("notice").textContent = payload.notice;
  } catch (error) {
    if (currentRequest === requestCounter) {
      container.textContent = `Não foi possível carregar as sugestões: ${error.message}`;
      byId("notice").textContent = "Confira se a API local está em execução.";
    }
  } finally {
    if (currentRequest === requestCounter) byId("apply").disabled = false;
  }
}

async function loadMetrics() {
  try {
    const response = await fetch("/api/metrics");
    if (!response.ok) throw new Error(`API retornou ${response.status}`);
    const metrics = await response.json();
    byId("ndcg").textContent = metrics.test.model.ndcg_at_5.toFixed(3);
    byId("baseline").textContent = metrics.test.rating_baseline.ndcg_at_5.toFixed(3);
    byId("hit").textContent = percentage(metrics.test.model.hit_rate_at_5);
    byId("sessions").textContent = metrics.test.model.sessions;
  } catch (error) {
    byId("notice").textContent = `Métricas indisponíveis: ${error.message}`;
  }
}

document.querySelectorAll("#budgets button").forEach((button) => {
  button.addEventListener("click", () => updateBudgetButtons(button));
  button.setAttribute("aria-pressed", String(button.dataset.value === "2"));
});
byId("eta").addEventListener("input", () => {
  byId("etaLabel").textContent = `Até ${byId("eta").value} min`;
});
byId("diversity").addEventListener("input", () => {
  byId("divLabel").textContent = percentage(byId("diversity").value);
});
byId("apply").addEventListener("click", loadRecommendations);

loadMetrics();
loadRecommendations();
loadDiversityStudy();
