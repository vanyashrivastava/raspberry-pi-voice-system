/* Owner: Nicole
   Responsibility: Light JS for polling alerts and wiring acknowledge buttons.
   Notes:
     - Keep logic small to reduce CPU usage on the Pi.
     - Uses fetch() to call the backend API endpoints.
*/

async function fetchAlerts() {
  try {
    const res = await fetch('/api/alerts');
    if (!res.ok) return [];
    return await res.json();
  } catch (e) {
    console.error('fetchAlerts error', e);
    return [];
  }
}

async function refreshTable() {
  const alerts = await fetchAlerts();
  const tbody = document.querySelector('#alerts-table tbody');
  tbody.innerHTML = '';
  if (!alerts || alerts.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5">No alerts yet.</td></tr>';
    return;
  }
  for (const a of alerts) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${a.ts || ''}</td><td>${a.source || ''}</td><td>${a.scam_probability || ''}</td><td>${(a.type||'')+' '+(a.note||'')}</td><td><button class="ack" data-id="${a.message_id||''}">Acknowledge</button></td>`;
    tbody.appendChild(tr);
  }
  attachButtons();
}

function attachButtons() {
  document.querySelectorAll('button.ack').forEach(btn => {
    btn.onclick = async () => {
      const id = btn.dataset.id;
      await fetch('/api/ack', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({message_id: id}) });
      btn.textContent = 'Acked';
      btn.disabled = true;
    };
  });
}

// Poll every 5 seconds
setInterval(refreshTable, 5000);
// Initial load
refreshTable();
