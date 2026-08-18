"""
discord_bot/bot.py
==================
Discord Bot untuk deteksi cyberbullying dengan sistem moderasi berbasis SEVERITY.

Sistem Severity:
    non_bullying → tidak ada tindakan
    mild         → Warn + DM + hapus pesan
    moderate     → Timeout (30 menit, configurable) + DM + hapus pesan
    severe       → Timeout panjang + DM + notif mod channel wajib

Sistem Eskalasi (kumulatif):
    Pelanggaran ke-1   → aksi sesuai severity
    Pelanggaran ke-3+  → timeout 24 jam
    Pelanggaran ke-5+  → kick

Slash Commands:
    /status            — status bot dan statistik server
    /violations @user  — lihat riwayat pelanggaran (mod)
    /reset @user       — reset pelanggaran user (admin)
    /appeal <alasan>   — user ajukan banding atas tindakan
    /toggle            — aktifkan/nonaktifkan deteksi (admin)
    /metrics           — metrik performa bot (mod)
    /mod_review        — lihat antrian review moderator (mod)

Action Config via YAML:
    Edit configs/moderation_config.yaml untuk mengubah
    threshold, durasi, dan behavior bot tanpa mengubah kode ini.

Cara jalankan:
    cd discord_bot
    python bot.py

Atau dari root project:
    python -m discord_bot.bot
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands, tasks

# ──────────────────────────────────────────────────────────────
# Path setup
# ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from .config import Config
from .database import Database
from .metrics import metrics
from .preprocessor import preprocess, compute_sarcasm_confidence_boost

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bot")


# ──────────────────────────────────────────────────────────────
# Bot Setup
# ──────────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.guilds = True

bot = commands.Bot(command_prefix="!", intents=intents)
tree = bot.tree

# Global state
db = Database()
detection_enabled: dict[int, bool] = {}   # guild_id → bool
http_session: aiohttp.ClientSession = None
n8n_healthy = True
last_n8n_failure_time = 0.0


# ──────────────────────────────────────────────────────────────
# Background Tasks
# ──────────────────────────────────────────────────────────────
@tasks.loop(minutes=30)
async def auto_save_metrics():
    """
    Background task: simpan snapshot metrik performa ke database setiap 30 menit.

    Memastikan tabel `performance_logs` selalu terisi data fresh sehingga
    n8n dapat mengambil data evaluasi sistem kapan saja tanpa harus menunggu
    moderator menjalankan command /metrics di Discord.
    """
    try:
        snap = metrics.snapshot()
        # guild_id None berarti snapshot global (bukan per-server)
        db.save_performance_snapshot(guild_id=None, snapshot=snap)
        logger.debug(
            f"[auto_save_metrics] Snapshot tersimpan — "
            f"pesan={snap['total_messages']:,} | "
            f"latency_p95={snap.get('total_p95', '-')} ms | "
            f"uptime={snap['uptime_str']}"
        )
    except Exception as e:
        logger.error(f"[auto_save_metrics] Gagal menyimpan snapshot: {e}", exc_info=True)


@auto_save_metrics.before_loop
async def before_auto_save():
    """Tunggu bot siap sebelum memulai loop pertama."""
    await bot.wait_until_ready()


# ──────────────────────────────────────────────────────────────
# Helpers — Colours & Embeds
# ──────────────────────────────────────────────────────────────
COLOR_WARN      = discord.Color.orange()
COLOR_MODERATE  = discord.Color.red()
COLOR_SEVERE    = discord.Color.dark_red()
COLOR_FLAG      = discord.Color.yellow()
COLOR_OK        = discord.Color.green()
COLOR_INFO      = discord.Color.blurple()
COLOR_MUTE      = discord.Color.red()
COLOR_KICK      = discord.Color.dark_red()

# Emoji label per severity
SEVERITY_EMOJI = {
    "non_bullying": "✅",
    "mild":         "⚠️",
    "moderate":     "🔇",
    "severe":       "🚫",
}

SEVERITY_LABEL = {
    "non_bullying": "Non-Bullying",
    "mild":         "Ringan (Mild)",
    "moderate":     "Sedang (Moderate)",
    "severe":       "Parah (Severe)",
}


# ──────────────────────────────────────────────────────────────
# Badword Filter — kata kasar eksplisit yang langsung kena ACTION
# tanpa tergantung confidence model
# ──────────────────────────────────────────────────────────────
import re

BADWORDS = {
    # Indonesia
    "kontol", "memek", "ngentot", "ngewe", "pepek", "jembut",
    "bajingan", "sundala", "telaso", "keparat", "brengsek",
    "tolol", "goblok", "bego", "idiot", "dungu",
    "tai", "tahi", "pantek", "puki", "pukimak",
    "asu", "jancok", "jancuk", "cok", "dancok",
    "bangsat", "kampret", "kampang", "kimak",
    "ngtd", "ngntd", "pepek", "ppk", "mmk", "kontl", "asuw", "asu", "ngews",
    "ngeww", "gblg", "gblog", "njing", "kntl", "anjg", "anj", 
    # English
    "fuck", "shit", "bitch", "asshole", "dick",
    "cunt", "bastard", "nigger", "faggot", "retard", "nigga",
}

# ──────────────────────────────────────────────────────────────
# Positive Whitelist — kata/frasa yang menunjukkan konteks positif
# ──────────────────────────────────────────────────────────────
POSITIVE_WORDS = {
    "keren", "bagus", "mantap", "mantul", "kece", "sip", "top",
    "hebat", "luar biasa", "kece badai", "cakep", "cantik", "ganteng",
    "pintar", "cerdas", "rajin", "baik", "ramah", "manis",
    "lucu", "imut", "menggemaskan", "menarik", "bercanda",
    "halo", "hai", "hey", "hei", "selamat", "salam",
    "pagi", "siang", "sore", "malam", "gg", "good game",
    "makasih", "terimakasih", "terima kasih", "thanks", "thx",
    "oke", "ok", "siap", "sip", "noted", "nice",
    "wah", "wow", "kaget", "asik", "asek",
    "demam", "sakit", "sehat", "cape", "capek", "lelah",
    "lapar", "kenyang", "ngantuk", "tidur", "bangun",
    "wkwk", "wkwkwk", "haha", "hihi", "lol", "lmao",
    "xixi", "kwkw", "hehe", "tes", "test",
    "pendukung", "fans", "suporter"
}

SAFE_PATTERNS = [
    r'^(halo|hai|hey|hei|hi)\b',
    r'^(pagi|siang|sore|malam)\b',
    r'^(gg|good game|nice|good)\b',
    r'^(ok|oke|siap|noted|sip)\b',
    r'^(makasih|thanks|thx|ty)\b',
    r'^(wkwk|haha|hihi|lol|xixi)\b',
]


def contains_badword(text: str) -> bool:
    """Cek apakah teks mengandung kata kasar dari daftar."""
    words = re.findall(r'[a-zA-Z]+', text.lower())
    return any(w in BADWORDS for w in words)


def is_likely_safe(text: str) -> bool:
    """
    Cek apakah pesan kemungkinan besar AMAN (false positive guard).
    Returns True jika pesan AMAN dan sebaiknya di-skip dari deteksi.
    """
    text_lower = text.lower().strip()
    words = text_lower.split()

    # Guard untuk pembahasan teknis bot/dataset/model (meta-talk)
    meta_words = {"dataset", "detect", "deteksi", "database"}
    word_set = set(re.findall(r'[a-zA-Z]+', text_lower))
    if (word_set & meta_words) and not contains_badword(text):
        return True

    for pattern in SAFE_PATTERNS:
        if re.match(pattern, text_lower):
            return True

    if len(words) <= 4:
        has_positive = bool(word_set & POSITIVE_WORDS)
        has_badword  = contains_badword(text)
        if has_positive and not has_badword:
            return True

    return False


# ──────────────────────────────────────────────────────────────
# Severity Routing Helpers
# ──────────────────────────────────────────────────────────────

def get_escalated_action(violation_count: int) -> dict | None:
    """
    Cek apakah ada aturan eskalasi yang berlaku berdasarkan
    jumlah pelanggaran kumulatif user.

    Return: dict action override atau None jika tidak ada eskalasi.
    """
    rules = Config.get_escalation_rules()
    if not rules:
        return None

    # Urutkan berdasarkan min_violations desc, ambil yang paling cocok
    applicable = [r for r in rules if violation_count >= r.get("min_violations", 1)]
    if not applicable:
        return None

    return sorted(applicable, key=lambda r: r["min_violations"], reverse=True)[0]


def _format_duration(minutes: int) -> str:
    """Format durasi menit ke string yang mudah dibaca."""
    if minutes < 60:
        return f"{minutes} menit"
    elif minutes < 1440:
        return f"{minutes // 60} jam"
    else:
        return f"{minutes // 1440} hari {(minutes % 1440) // 60} jam"


# ──────────────────────────────────────────────────────────────
# Helpers — n8n / API
# ──────────────────────────────────────────────────────────────
async def call_n8n(text: str, user_id: int, guild_id: int, message_id: int) -> dict | None:
    """
    Kirim pesan ke n8n webhook untuk diproses.
    n8n kemudian memanggil FastAPI /predict dan mengembalikan hasilnya.
    """
    global n8n_healthy, last_n8n_failure_time
    import time

    if not n8n_healthy and (time.time() - last_n8n_failure_time) < 60:
        return None

    payload = {
        "text":       text,
        "user_id":    str(user_id),
        "guild_id":   str(guild_id),
        "message_id": str(message_id),
        "lang":       "id",
    }
    _t0 = time.perf_counter()
    try:
        logger.debug(f"[n8n] POST → {Config.N8N_WEBHOOK_URL}")
        async with http_session.post(
            Config.N8N_WEBHOOK_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=15),   # naik dari 3s → 15s
        ) as resp:
            _latency_ms = (time.perf_counter() - _t0) * 1000
            if resp.status == 200:
                n8n_healthy = True
                metrics.record_n8n_call(_latency_ms, success=True)
                logger.debug(f"[n8n] Response OK ({_latency_ms:.0f} ms)")
                # Coba parse JSON — body bisa kosong jika workflow n8n belum dikonfigurasi
                # dengan benar (node Respond To Webhook tidak terhubung)
                raw_body = await resp.text()
                if not raw_body or not raw_body.strip():
                    logger.error(
                        "[n8n] Response HTTP 200 tapi body KOSONG.\n"
                        "      Penyebab: Node 'Respond to Webhook' di workflow n8n tidak\n"
                        "      terhubung ke akhir alur, atau workflow berjalan dengan\n"
                        "      responseMode bukan 'responseNode'.\n"
                        "      Solusi: Buka n8n UI → workflow cyberbully → pastikan node\n"
                        "      'Respond — Ignore/Flag/Action' terhubung dan aktif."
                    )
                    n8n_healthy = False
                    last_n8n_failure_time = time.time()
                    return None
                try:
                    import json as _json
                    data = _json.loads(raw_body)
                    logger.debug(f"[n8n] JSON parsed: {data}")
                    return data
                except _json.JSONDecodeError:
                    logger.error(
                        f"[n8n] Response HTTP 200 tapi bukan JSON valid.\n"
                        f"      Body: {raw_body[:300]}\n"
                        f"      Cek konfigurasi node 'Respond to Webhook' di n8n."
                    )
                    n8n_healthy = False
                    last_n8n_failure_time = time.time()
                    return None
            else:
                body = await resp.text()
                logger.warning(
                    f"[n8n] Webhook error: HTTP {resp.status} — {body[:200]}\n"
                    f"      URL: {Config.N8N_WEBHOOK_URL}\n"
                    f"      Pastikan workflow aktif (toggle Active di n8n UI)."
                )
                n8n_healthy = False
                last_n8n_failure_time = time.time()
                metrics.record_n8n_call(_latency_ms, success=False)
                return None
    except asyncio.TimeoutError:
        _latency_ms = (time.perf_counter() - _t0) * 1000
        logger.error(
            f"[n8n] Webhook timeout (>{_latency_ms/1000:.0f}s)\n"
            f"      URL: {Config.N8N_WEBHOOK_URL}\n"
            f"      Cek: 1) n8n sudah running? 2) workflow aktif? 3) FastAPI merespons?"
        )
        n8n_healthy = False
        last_n8n_failure_time = time.time()
        metrics.record_n8n_call(_latency_ms, success=False)
        return None
    except aiohttp.ClientConnectorError as e:
        _latency_ms = (time.perf_counter() - _t0) * 1000
        logger.error(
            f"[n8n] Tidak bisa terhubung ke {Config.N8N_WEBHOOK_URL}\n"
            f"      Error: {e}\n"
            f"      Cek N8N_WEBHOOK_URL di .env — apakah port n8n sudah benar?"
        )
        n8n_healthy = False
        last_n8n_failure_time = time.time()
        metrics.record_n8n_call(_latency_ms, success=False)
        return None
    except aiohttp.ClientError as e:
        _latency_ms = (time.perf_counter() - _t0) * 1000
        logger.error(f"[n8n] Webhook connection error: {e}")
        n8n_healthy = False
        last_n8n_failure_time = time.time()
        metrics.record_n8n_call(_latency_ms, success=False)
        return None


async def call_api_direct(text: str) -> dict | None:
    """Fallback: langsung panggil FastAPI /predict jika n8n tidak tersedia."""
    import time
    _t0 = time.perf_counter()
    try:
        async with http_session.post(
            f"{Config.API_URL}/predict",
            json={"text": text, "lang": "id"},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            _latency_ms = (time.perf_counter() - _t0) * 1000
            if resp.status == 200:
                metrics.record_api_call(_latency_ms, success=True)
                return await resp.json()
            else:
                logger.warning(f"API error: HTTP {resp.status}")
                metrics.record_api_call(_latency_ms, success=False)
                return None
    except Exception as e:
        _latency_ms = (time.perf_counter() - _t0) * 1000
        logger.error(f"API connection error: {e}")
        metrics.record_api_call(_latency_ms, success=False)
        metrics.record_error()
        return None


# ──────────────────────────────────────────────────────────────
# Helpers — Common
# ──────────────────────────────────────────────────────────────
async def get_mod_channel(guild: discord.Guild) -> discord.TextChannel | None:
    channel = guild.get_channel(Config.MOD_CHANNEL_ID)
    if channel is None:
        logger.warning(f"Channel moderasi (ID={Config.MOD_CHANNEL_ID}) tidak ditemukan di guild {guild.name}")
    return channel


async def send_dm(user: discord.Member, message: str):
    """Kirim DM ke user (abaikan jika DM dimatikan)."""
    try:
        await user.send(message)
    except discord.Forbidden:
        logger.info(f"Tidak bisa kirim DM ke {user} (DM dimatikan)")
    except Exception as e:
        logger.warning(f"Gagal kirim DM ke {user}: {e}")


async def delete_message_safe(message: discord.Message):
    """Hapus pesan dengan penanganan error."""
    try:
        await message.delete()
    except discord.NotFound:
        pass
    except discord.Forbidden:
        logger.warning("Bot tidak punya izin untuk menghapus pesan!")


async def apply_timeout(
    user: discord.Member,
    guild: discord.Guild,
    duration_minutes: int,
    reason: str,
) -> datetime | None:
    """
    Terapkan Discord Timeout ke user.
    Return: muted_until datetime atau None jika gagal.
    """
    timeout_duration = timedelta(minutes=duration_minutes)
    muted_until = datetime.now(timezone.utc) + timeout_duration

    try:
        await user.timeout(timeout_duration, reason=reason)
        db.set_muted_until(user.id, guild.id, muted_until)
        return muted_until
    except discord.Forbidden:
        logger.warning("Bot tidak punya izin untuk timeout member!")
        return None
    except Exception as e:
        logger.error(f"Gagal timeout {user}: {e}")
        return None


async def restore_message_via_webhook(
    guild: discord.Guild,
    channel_id: int | str | None,
    user_id: int | str,
    message_content: str,
) -> bool:
    """
    Kirim kembali pesan yang terhapus ke channel aslinya menggunakan Webhook
    dengan nama & avatar user asli.
    """
    if not channel_id or not message_content:
        logger.warning("[restore_message] channel_id atau message_content kosong")
        return False

    try:
        channel = guild.get_channel(int(channel_id))
        if not channel or not isinstance(channel, discord.TextChannel):
            logger.warning(f"[restore_message] Channel {channel_id} tidak ditemukan atau bukan TextChannel")
            return False

        member = guild.get_member(int(user_id))
        if not member:
            try:
                member = await guild.fetch_member(int(user_id))
            except Exception:
                member = None

        username = member.display_name if member else f"User ({user_id})"
        avatar_url = member.display_avatar.url if member else None

        try:
            webhooks = await channel.webhooks()
            webhook = next((w for w in webhooks if w.name == "Cyberbully-Restorer"), None)
            if not webhook:
                webhook = await channel.create_webhook(name="Cyberbully-Restorer")

            await webhook.send(
                content=f"{message_content}\n\n*(♻️ Pesan dipulihkan setelah peninjauan moderator)*",
                username=username,
                avatar_url=avatar_url,
            )
            logger.info(f"[restore_message] Pesan berhasil dipulihkan via Webhook di #{channel.name} untuk user {user_id}")
            return True
        except discord.Forbidden:
            logger.warning(
                f"[restore_message] Bot tidak memiliki izin 'Manage Webhooks' di #{channel.name}. "
                "Menggunakan fallback pesan biasa ..."
            )
            user_ref = member.mention if member else f"**{username}**"
            await channel.send(
                f"📢 **[Pesan Dipulihkan]** dari {user_ref}:\n"
                f"> {message_content}\n"
                f"*(♻️ Pesan dipulihkan setelah peninjauan moderator)*"
            )
            return True
    except Exception as e:
        logger.error(f"[restore_message] Gagal memulihkan pesan: {e}")
        return False



# ──────────────────────────────────────────────────────────────
# discord.ui.View — Mod Review Interaktif
# ──────────────────────────────────────────────────────────────
class ModReviewView(discord.ui.View):
    """
    Tombol interaktif untuk moderator di channel mod-review.
    Muncul pada kasus severity 'severe' atau confidence rendah.
    """

    def __init__(self, review_id: int, user_id: int, guild_id: int):
        super().__init__(timeout=86400)  # tombol aktif 24 jam
        self.review_id = review_id
        self.user_id   = user_id
        self.guild_id  = guild_id

    async def _check_mod_permission(self, interaction: discord.Interaction) -> bool:
        """Hanya moderator yang bisa menekan tombol."""
        if not interaction.user.guild_permissions.manage_messages:
            await interaction.response.send_message(
                "❌ Hanya moderator yang bisa menggunakan tombol ini.",
                ephemeral=True,
            )
            return False
        return True

    @discord.ui.button(
        label="✅ Dismiss (Aman)",
        style=discord.ButtonStyle.success,
        custom_id="mod_review_dismiss",
    )
    async def btn_dismiss(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not await self._check_mod_permission(interaction):
            return

        await interaction.response.defer(ephemeral=True)

        db.resolve_review(
            review_id=self.review_id,
            reviewed_by=interaction.user.id,
            status="dismissed",
            note=f"Dismissed oleh {interaction.user}",
        )

        # Coba pulihkan pesan terhapus via webhook jika data tersedia
        review = db.get_review_by_id(self.review_id)
        restored = False
        if review and review.get("channel_id") and review.get("message_content"):
            restored = await restore_message_via_webhook(
                guild=interaction.guild,
                channel_id=review["channel_id"],
                user_id=self.user_id,
                message_content=review["message_content"],
            )

        # Update embed
        embed = interaction.message.embeds[0]
        embed.color = COLOR_OK
        res_text = "\n*(♻️ Pesan telah dipulihkan ke channel)*" if restored else ""
        embed.add_field(
            name="✅ Keputusan",
            value=f"**DISMISSED** oleh {interaction.user.mention}{res_text}\n_{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC_",
            inline=False,
        )
        await interaction.message.edit(embed=embed, view=None)
        await interaction.followup.send(
            f"✅ Kasus di-dismiss oleh {interaction.user.mention}." + (" Pesan telah dipulihkan via Webhook." if restored else ""),
            ephemeral=True
        )

    @discord.ui.button(
        label="⚠️ Warn User",
        style=discord.ButtonStyle.secondary,
        custom_id="mod_review_warn",
    )
    async def btn_warn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not await self._check_mod_permission(interaction):
            return

        await interaction.response.defer(ephemeral=True)

        guild  = interaction.guild
        member = guild.get_member(self.user_id)

        if member:
            violation_count = db.increment_violation(self.user_id, self.guild_id)
            await send_dm(
                member,
                f"⚠️ **Peringatan dari {guild.name}**\n\n"
                f"Moderator telah meninjau pesan kamu dan memutuskan memberikan **peringatan**.\n"
                f"Ini adalah pelanggaran ke-**{violation_count}** kamu.",
            )
            db.resolve_review(
                review_id=self.review_id,
                reviewed_by=interaction.user.id,
                status="approved",
                note=f"Warn oleh {interaction.user}",
            )
            db.log_action(
                user_id=self.user_id, guild_id=self.guild_id,
                action="mod_warn", action_tier="action",
            )

        embed = interaction.message.embeds[0]
        embed.color = COLOR_WARN
        embed.add_field(
            name="⚠️ Keputusan",
            value=f"**WARN** dikirim oleh {interaction.user.mention}",
            inline=False,
        )
        await interaction.message.edit(embed=embed, view=None)
        await interaction.followup.send(
            f"⚠️ Peringatan dikirim ke user.", ephemeral=True
        )

    @discord.ui.button(
        label="🔇 Timeout 30m",
        style=discord.ButtonStyle.danger,
        custom_id="mod_review_timeout",
    )
    async def btn_timeout(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not await self._check_mod_permission(interaction):
            return

        await interaction.response.defer(ephemeral=True)

        guild  = interaction.guild
        member = guild.get_member(self.user_id)

        if member:
            muted_until = await apply_timeout(
                member, guild, duration_minutes=30,
                reason=f"Timeout oleh moderator {interaction.user}"
            )
            violation_count = db.increment_violation(self.user_id, self.guild_id)

            if muted_until:
                await send_dm(
                    member,
                    f"🔇 **Kamu di-Timeout di {guild.name}**\n\n"
                    f"Moderator telah meninjau pesan kamu dan memutuskan "
                    f"timeout selama **30 menit**.\n"
                    f"Berakhir: <t:{int(muted_until.timestamp())}:f>",
                )
            db.resolve_review(
                review_id=self.review_id,
                reviewed_by=interaction.user.id,
                status="approved",
                note=f"Timeout 30m oleh {interaction.user}",
            )
            db.log_action(
                user_id=self.user_id, guild_id=self.guild_id,
                action="mod_timeout_30m", action_tier="action",
                violation_count=violation_count,
            )

        embed = interaction.message.embeds[0]
        embed.color = COLOR_MUTE
        embed.add_field(
            name="🔇 Keputusan",
            value=f"**TIMEOUT 30m** oleh {interaction.user.mention}",
            inline=False,
        )
        await interaction.message.edit(embed=embed, view=None)
        await interaction.followup.send(
            f"🔇 User di-timeout 30 menit.", ephemeral=True
        )


class AppealReviewView(discord.ui.View):
    """Tombol interaktif untuk moderator mereview appeal dari user."""

    def __init__(self, appeal_id: int, appellant_id: int):
        super().__init__(timeout=172800)  # 48 jam
        self.appeal_id    = appeal_id
        self.appellant_id = appellant_id

    async def _check_mod(self, interaction: discord.Interaction) -> bool:
        if not interaction.user.guild_permissions.manage_messages:
            await interaction.response.send_message(
                "❌ Hanya moderator yang bisa mereview appeal.", ephemeral=True
            )
            return False
        return True

    @discord.ui.button(label="✅ Approve Appeal", style=discord.ButtonStyle.success)
    async def btn_approve(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not await self._check_mod(interaction):
            return

        await interaction.response.defer(ephemeral=True)

        db.resolve_appeal(
            appeal_id=self.appeal_id,
            reviewed_by=interaction.user.id,
            status="approved",
            note="Disetujui oleh moderator",
        )

        # Notifikasi ke user
        member = interaction.guild.get_member(self.appellant_id)
        if member:
            await send_dm(
                member,
                f"✅ **Banding Kamu Disetujui — {interaction.guild.name}**\n\n"
                f"Moderator telah meninjau kasusmu dan menyetujui banding kamu.\n"
                f"Tindakan yang diterima akan dibatalkan jika memungkinkan.\n"
                f"Terima kasih atas laporanmu!",
            )
            # Cabut timeout jika masih aktif
            is_muted, _ = db.is_muted(self.appellant_id, interaction.guild_id)
            if is_muted:
                try:
                    await member.timeout(None, reason="Appeal disetujui oleh moderator")
                    db.set_muted_until(member.id, interaction.guild_id,
                                       datetime.now(timezone.utc))
                except Exception as e:
                    logger.warning(f"Gagal cabut timeout untuk appeal: {e}")

        # Coba pulihkan pesan terhapus via webhook dari log tindakan terbaru user ini
        restored = False
        logs = db.get_user_logs(self.appellant_id, interaction.guild_id, limit=5)
        if logs:
            target_log = next((l for l in logs if l.get("message_content") and l.get("channel_id")), None)
            if target_log:
                restored = await restore_message_via_webhook(
                    guild=interaction.guild,
                    channel_id=target_log["channel_id"],
                    user_id=self.appellant_id,
                    message_content=target_log["message_content"],
                )

        embed = interaction.message.embeds[0]
        embed.color = COLOR_OK
        res_text = "\n*(♻️ Pesan telah dipulihkan ke channel)*" if restored else ""
        embed.add_field(
            name="✅ Keputusan",
            value=f"**DISETUJUI** oleh {interaction.user.mention}{res_text}",
            inline=False,
        )
        await interaction.message.edit(embed=embed, view=None)
        await interaction.followup.send(
            "✅ Appeal disetujui." + (" Pesan terhapus telah dipulihkan via Webhook." if restored else ""),
            ephemeral=True
        )

    @discord.ui.button(label="❌ Reject Appeal", style=discord.ButtonStyle.danger)
    async def btn_reject(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not await self._check_mod(interaction):
            return

        await interaction.response.defer(ephemeral=True)

        db.resolve_appeal(
            appeal_id=self.appeal_id,
            reviewed_by=interaction.user.id,
            status="rejected",
            note="Ditolak oleh moderator",
        )

        member = interaction.guild.get_member(self.appellant_id)
        if member:
            await send_dm(
                member,
                f"❌ **Banding Kamu Ditolak — {interaction.guild.name}**\n\n"
                f"Moderator telah meninjau kasusmu dan memutuskan bahwa "
                f"tindakan yang diberikan tetap berlaku.\n"
                f"Harap patuhi aturan server ke depannya.",
            )

        embed = interaction.message.embeds[0]
        embed.color = discord.Color.dark_red()
        embed.add_field(
            name="❌ Keputusan",
            value=f"**DITOLAK** oleh {interaction.user.mention}",
            inline=False,
        )
        await interaction.message.edit(embed=embed, view=None)
        await interaction.followup.send("❌ Appeal ditolak.", ephemeral=True)


# ──────────────────────────────────────────────────────────────
# Moderation Handlers — Per Severity
# ──────────────────────────────────────────────────────────────

async def handle_mild(
    message: discord.Message,
    confidence: float,
    severity: str,
    violation_count: int,
):
    """Severity MILD: Warn + Hapus pesan + DM."""
    user  = message.author
    guild = message.guild

    await delete_message_safe(message)

    cfg = Config.get_action_config("mild")
    dm_msg = cfg.get("dm_message", "").format(
        guild_name=guild.name,
        duration="N/A",
    ).strip()

    if cfg.get("dm") and dm_msg:
        await send_dm(user, dm_msg)
    else:
        await send_dm(
            user,
            f"⚠️ **Peringatan dari {guild.name}**\n\n"
            f"Pesan kamu telah dihapus karena terdeteksi mengandung "
            f"konten yang bisa menyinggung (*mild bullying*).\n\n"
            f"🔢 Pelanggaran ke-**{violation_count}** kamu.\n"
            f"Jika kamu merasa ini keliru, gunakan `/appeal` di server.",
        )

    if cfg.get("send_to_mod_channel", False):
        mod_channel = await get_mod_channel(guild)
        if mod_channel:
            embed = _build_action_embed(
                title="⚠️ Deteksi — MILD",
                description=f"{user.mention} mendapat **peringatan** (pelanggaran ke-{violation_count}).",
                color=COLOR_WARN,
                user=user,
                message_content=message.content,
                confidence=confidence,
                severity=severity,
                violation_count=violation_count,
            )
            await mod_channel.send(embed=embed)

    logger.info(f"MILD | user={user} | guild={guild.name} | violation={violation_count}")


async def handle_moderate(
    message: discord.Message,
    confidence: float,
    severity: str,
    violation_count: int,
):
    """Severity MODERATE: Timeout + Hapus pesan + DM + notif mod channel."""
    user  = message.author
    guild = message.guild

    await delete_message_safe(message)

    cfg              = Config.get_action_config("moderate")
    duration_minutes = cfg.get("duration_minutes", 30)
    dur_str          = _format_duration(duration_minutes)

    muted_until = await apply_timeout(
        user, guild,
        duration_minutes=duration_minutes,
        reason=f"Cyberbullying moderate (pelanggaran ke-{violation_count}, conf={confidence:.2f})",
    )

    dm_msg = cfg.get("dm_message", "").format(
        guild_name=guild.name,
        duration=dur_str,
    ).strip()

    if cfg.get("dm") and dm_msg:
        await send_dm(user, dm_msg)
    elif muted_until:
        await send_dm(
            user,
            f"🔇 **Kamu di-Timeout di {guild.name}**\n\n"
            f"Pesan kamu dihapus karena cyberbullying tingkat sedang (pelanggaran ke-{violation_count}).\n\n"
            f"⏳ Kamu di-timeout selama **{dur_str}** "
            f"(sampai <t:{int(muted_until.timestamp())}:f>).\n\n"
            f"Gunakan `/appeal` jika kamu merasa ini keliru.",
        )

    # Kirim ke mod channel
    mod_channel = await get_mod_channel(guild)
    if mod_channel:
        embed = _build_action_embed(
            title="🔇 Deteksi — MODERATE",
            description=(
                f"{user.mention} di-**timeout {dur_str}** "
                f"(pelanggaran ke-{violation_count})."
                + (f"\nBerakhir: <t:{int(muted_until.timestamp())}:f>" if muted_until else "")
            ),
            color=COLOR_MODERATE,
            user=user,
            message_content=message.content,
            confidence=confidence,
            severity=severity,
            violation_count=violation_count,
        )
        await mod_channel.send(embed=embed)

    logger.info(f"MODERATE | user={user} | guild={guild.name} | duration={duration_minutes}m")


async def handle_severe(
    message: discord.Message,
    confidence: float,
    severity: str,
    violation_count: int,
):
    """
    Severity SEVERE: Timeout panjang + Hapus pesan + DM
    + kirim embed interaktif ke mod channel untuk review.
    """
    user  = message.author
    guild = message.guild

    await delete_message_safe(message)

    cfg              = Config.get_action_config("severe")
    duration_minutes = cfg.get("duration_minutes", 480)
    dur_str          = _format_duration(duration_minutes)
    need_approval    = cfg.get("require_mod_approval", False)

    muted_until = None
    if not need_approval:
        # Langsung eksekusi timeout
        muted_until = await apply_timeout(
            user, guild,
            duration_minutes=duration_minutes,
            reason=f"Cyberbullying severe (conf={confidence:.2f})",
        )

    dm_msg = cfg.get("dm_message", "").format(
        guild_name=guild.name,
        duration=dur_str,
    ).strip()

    if cfg.get("dm") and dm_msg:
        await send_dm(user, dm_msg)
    else:
        await send_dm(
            user,
            f"🚫 **Tindakan Keras di {guild.name}**\n\n"
            f"Pesan kamu mengandung cyberbullying tingkat parah dan telah dihapus.\n"
            + (f"Kamu di-timeout selama **{dur_str}**.\n" if muted_until else "")
            + "\nKasus kamu sedang ditinjau oleh moderator.\n"
              "Gunakan `/appeal` jika kamu merasa ini keliru.",
        )

    # Tambah ke antrian mod review
    review_id = db.add_to_review_queue(
        guild_id=guild.id,
        user_id=user.id,
        message_content=message.content[:500],
        confidence=confidence,
        severity=severity,
        action_tier="action",
        channel_id=message.channel.id,
        message_id=message.id,
    )

    # Kirim embed interaktif ke mod channel
    mod_channel = await get_mod_channel(guild)
    if mod_channel:
        mod_review_cfg = Config.get_mod_review_config()
        use_buttons    = mod_review_cfg.get("interactive_buttons", True)

        embed = _build_action_embed(
            title="🚫 Deteksi — SEVERE (Review Diperlukan)",
            description=(
                f"{user.mention} mendapat tindakan **severe** (pelanggaran ke-{violation_count}).\n"
                + (f"Timeout **{dur_str}** telah diterapkan otomatis." if muted_until else
                   "⏳ Menunggu keputusan moderator untuk eksekusi tindakan.")
            ),
            color=COLOR_SEVERE,
            user=user,
            message_content=message.content,
            confidence=confidence,
            severity=severity,
            violation_count=violation_count,
        )
        embed.add_field(
            name="🆔 Review ID",
            value=f"`#{review_id}`",
            inline=True,
        )

        view = ModReviewView(
            review_id=review_id,
            user_id=user.id,
            guild_id=guild.id,
        ) if use_buttons else None

        embed_msg = await mod_channel.send(embed=embed, view=view)

        # Update embed_message_id di database
        db.update_review_embed_id(review_id, embed_msg.id)

    logger.info(f"SEVERE | user={user} | guild={guild.name} | review_id={review_id}")


async def handle_escalated_action(
    message: discord.Message,
    confidence: float,
    severity: str,
    violation_count: int,
    escalation_rule: dict,
):
    """
    Handle tindakan yang di-eskalasi karena pelanggaran berulang.
    Override action dari escalation_rules di moderation_config.yaml.
    """
    user  = message.author
    guild = message.guild

    await delete_message_safe(message)

    action   = escalation_rule.get("action_override", "timeout")
    duration = escalation_rule.get("duration_minutes", 1440)
    dur_str  = _format_duration(duration)

    if action == "timeout":
        muted_until = await apply_timeout(
            user, guild,
            duration_minutes=duration,
            reason=f"Eskalasi pelanggaran berulang (ke-{violation_count})",
        )
        await send_dm(
            user,
            f"🔒 **Eskalasi Tindakan di {guild.name}**\n\n"
            f"Karena kamu telah melanggar aturan sebanyak **{violation_count} kali**, "
            f"kamu mendapat **timeout selama {dur_str}**.\n\n"
            f"Harap patuhi aturan server.\nGunakan `/appeal` jika kamu merasa ini keliru.",
        )

    elif action == "kick":
        await send_dm(
            user,
            f"🚫 **Kamu di-Kick dari {guild.name}**\n\n"
            f"Kamu telah melanggar aturan sebanyak **{violation_count} kali**. "
            f"Kamu di-kick dari server.\n\n"
            f"Kamu masih bisa bergabung kembali dengan link invite, "
            f"namun harap patuhi aturan.",
        )
        try:
            await guild.kick(user, reason=f"Eskalasi pelanggaran ke-{violation_count}")
        except discord.Forbidden:
            logger.warning("Bot tidak punya izin kick!")

    mod_channel = await get_mod_channel(guild)
    if mod_channel:
        embed = _build_action_embed(
            title=f"⚡ Eskalasi — {action.upper()} (Pelanggaran ke-{violation_count})",
            description=(
                f"{user.mention} mendapat tindakan eskalasi karena **{violation_count} pelanggaran**."
            ),
            color=COLOR_KICK,
            user=user,
            message_content=message.content,
            confidence=confidence,
            severity=severity,
            violation_count=violation_count,
        )
        await mod_channel.send(embed=embed)

    logger.info(f"ESCALATED_{action.upper()} | user={user} | violation={violation_count}")


async def handle_flag(message: discord.Message, confidence: float, severity: str):
    """Flag ke channel moderasi untuk review manual — tidak ada tindakan otomatis."""
    user  = message.author
    guild = message.guild

    # Tambah ke antrian mod review
    review_id = db.add_to_review_queue(
        guild_id=guild.id,
        user_id=user.id,
        message_content=message.content[:500],
        confidence=confidence,
        severity=severity,
        action_tier="flag",
        channel_id=message.channel.id,
        message_id=message.id,
    )

    mod_channel = await get_mod_channel(guild)
    if mod_channel:
        embed = discord.Embed(
            title=f"🔍 Flagged untuk Review — {SEVERITY_LABEL.get(severity, severity)}",
            description=(
                f"Pesan dari {user.mention} ditandai untuk **review manual** moderator.\n"
                f"Confidence di zona abu-abu ({confidence*100:.1f}%), "
                f"bot tidak mengambil tindakan otomatis."
            ),
            color=COLOR_FLAG,
        )
        embed.set_author(name=str(user), icon_url=user.display_avatar.url)
        embed.add_field(name="👤 User",        value=user.mention, inline=True)
        embed.add_field(name="📊 Confidence",  value=f"{confidence*100:.1f}%", inline=True)
        embed.add_field(name="🎯 Severity",    value=SEVERITY_LABEL.get(severity, severity), inline=True)
        embed.add_field(
            name="💬 Pesan",
            value=f"[Klik untuk lihat]({message.jump_url})\n"
                  f"```{message.content[:200]}{'...' if len(message.content)>200 else ''}```",
            inline=False,
        )
        embed.add_field(
            name="🆔 Review ID",
            value=f"`#{review_id}`",
            inline=True,
        )
        embed.set_footer(text=f"User ID: {user.id} | Channel: #{message.channel.name}")
        embed.timestamp = datetime.now(timezone.utc)

        view = ModReviewView(
            review_id=review_id,
            user_id=user.id,
            guild_id=guild.id,
        )
        flag_msg = await mod_channel.send(embed=embed, view=view)
        db.update_review_embed_id(review_id, flag_msg.id)

    logger.info(f"FLAG | user={user} | guild={guild.name} | conf={confidence:.3f}")


def _build_action_embed(
    title: str,
    description: str,
    color: discord.Color,
    user: discord.Member,
    message_content: str,
    confidence: float,
    severity: str,
    violation_count: int,
) -> discord.Embed:
    """Helper untuk membangun embed tindakan moderasi."""
    embed = discord.Embed(title=title, description=description, color=color)
    embed.set_author(name=str(user), icon_url=user.display_avatar.url)
    embed.add_field(name="👤 User",          value=user.mention, inline=True)
    embed.add_field(name="🔢 Pelanggaran",   value=str(violation_count), inline=True)
    embed.add_field(name="📊 Confidence",    value=f"{confidence*100:.1f}%", inline=True)
    embed.add_field(
        name="🎯 Severity",
        value=f"{SEVERITY_EMOJI.get(severity, '❓')} {SEVERITY_LABEL.get(severity, severity)}",
        inline=True,
    )
    embed.add_field(
        name="💬 Pesan",
        value=f"```{message_content[:200]}{'...' if len(message_content)>200 else ''}```",
        inline=False,
    )
    embed.set_footer(text=f"User ID: {user.id}")
    embed.timestamp = datetime.now(timezone.utc)
    return embed


# ──────────────────────────────────────────────────────────────
# Events
# ──────────────────────────────────────────────────────────────
@bot.event
async def on_ready():
    global http_session
    http_session = aiohttp.ClientSession()

    logger.info(f"Bot online: {bot.user} (ID: {bot.user.id})")
    logger.info(f"Terhubung ke {len(bot.guilds)} server")
    logger.info(f"Severity thresholds: {Config.get_severity_thresholds()}")

    try:
        synced = await tree.sync()
        logger.info(f"Slash commands synced: {len(synced)}")
    except Exception as e:
        logger.error(f"Gagal sync slash commands: {e}")

    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.watching,
            name="pesan untuk cyberbullying 🛡️",
        )
    )

    # Mulai background task auto-save metrics untuk n8n
    if not auto_save_metrics.is_running():
        auto_save_metrics.start()
        logger.info("[auto_save_metrics] Background task dimulai (interval: 30 menit).")


@bot.event
async def on_message(message: discord.Message):
    """Main listener: periksa setiap pesan untuk cyberbullying."""
    import time
    _msg_start = time.perf_counter()

    # Abaikan pesan dari bot
    if Config.IGNORE_BOTS and message.author.bot:
        return

    # Abaikan DM
    if not message.guild:
        return

    # Cek apakah deteksi aktif untuk server ini
    guild_id = message.guild.id
    if not detection_enabled.get(guild_id, True):
        await bot.process_commands(message)
        return

    # Abaikan prefix command
    if any(message.content.startswith(p) for p in Config.EXEMPT_PREFIXES):
        await bot.process_commands(message)
        return

    # ── Preprocessing via preprocessor.py ────────────────────
    raw_text = message.content.strip()
    prep     = preprocess(raw_text)

    if prep.is_empty:
        await bot.process_commands(message)
        return

    cleaned_text = prep.text

    # Abaikan pesan sangat pendek (1 kata, <10 char) yang bukan badword
    words_in_text = cleaned_text.split()
    if len(words_in_text) <= 1 and len(cleaned_text) < 10 and not contains_badword(cleaned_text):
        await bot.process_commands(message)
        return

    # ── False Positive Guard ──────────────────────────────────
    if is_likely_safe(cleaned_text) and not contains_badword(cleaned_text):
        logger.debug(f"SAFE_SKIP | user={message.author} | text={cleaned_text[:60]}")
        await bot.process_commands(message)
        return

    # ── Panggil n8n webhook ───────────────────────────────────
    used_fallback = False
    result = await call_n8n(cleaned_text, message.author.id, guild_id, message.id)

    # Fallback ke API langsung jika n8n gagal
    if result is None:
        logger.warning("n8n tidak merespons, mencoba langsung ke API ...")
        result = await call_api_direct(cleaned_text)
        if result is not None:
            used_fallback = True

    if result is None:
        logger.error("API tidak tersedia, melewati deteksi untuk pesan ini.")
        metrics.record_error()
        await bot.process_commands(message)
        return

    # ── Proses hasil prediksi ─────────────────────────────────
    label      = result.get("label", "non-bully")
    confidence = float(result.get("confidence_bully", 0.0))
    tier       = result.get("action_tier", "ignore")
    severity   = result.get("severity", "non_bullying")

    # Fallback: jika severity tidak dikembalikan oleh n8n/API tapi label adalah bully,
    # hitung secara lokal berdasarkan threshold.
    if severity == "non_bullying" and label == "bully":
        _thresholds = Config.get_severity_thresholds()
        if confidence >= _thresholds.get("severe", 0.92):
            severity = "severe"
        elif confidence >= _thresholds.get("moderate", 0.80):
            severity = "moderate"
        elif confidence >= _thresholds.get("mild", 0.70):
            severity = "mild"


    # ── Auxiliary Signal: boost confidence jika ada sinyal sarkasme ──
    if prep.has_sarcasm_signal or prep.punctuation_excess:
        old_conf   = confidence
        confidence = compute_sarcasm_confidence_boost(prep, confidence)
        if confidence != old_conf:
            logger.info(
                f"SARCASM_BOOST | conf {old_conf:.3f}→{confidence:.3f} | "
                f"sarcasm={prep.has_sarcasm_signal} | punct={prep.punctuation_excess}"
            )
            # Re-compute severity berdasarkan confidence yang sudah di-boost
            thresholds = Config.get_severity_thresholds()
            if label == "bully":
                if confidence >= thresholds.get("severe", 0.92):
                    severity = "severe"
                elif confidence >= thresholds.get("moderate", 0.80):
                    severity = "moderate"
                elif confidence >= thresholds.get("mild", 0.70):
                    severity = "mild"

    # ── Confidence Guard — threshold tinggi untuk pesan pendek ──
    words_count = len(cleaned_text.split())
    if tier == "action" and words_count <= 5 and confidence < 0.92:
        tier     = "flag"
        severity = "mild" if severity in ("moderate", "severe") else severity
        logger.info(
            f"CONF_GUARD | teks pendek ({words_count} kata) conf={confidence:.3f} "
            f"→ downgrade action→flag | text={cleaned_text[:60]}"
        )
    elif tier == "action" and words_count <= 3 and confidence < 0.97:
        tier     = "flag"
        severity = "mild" if severity in ("moderate", "severe") else severity
        logger.info(
            f"CONF_GUARD | teks sangat pendek ({words_count} kata) conf={confidence:.3f} "
            f"→ downgrade action→flag | text={cleaned_text[:60]}"
        )

    # ── Badword override → selalu severe ─────────────────────
    if contains_badword(cleaned_text):
        tier       = "action"
        label      = "bully"
        confidence = max(confidence, 0.95)
        severity   = "severe"
        logger.info(f"BADWORD | user={message.author} | text={cleaned_text[:50]}")

    # ── Log ke database ───────────────────────────────────────
    db.log_action(
        user_id=message.author.id,
        guild_id=guild_id,
        action=f"detected_{tier}",
        message_content=raw_text[:500],
        confidence=confidence,
        action_tier=tier,
        severity=severity,
        channel_id=message.channel.id,
        message_id=message.id,
    )

    # ── Routing berdasarkan severity ──────────────────────────
    if tier == "ignore" or severity == "non_bullying":
        pass  # Tidak ada tindakan

    elif tier == "flag":
        await handle_flag(message, confidence, severity)

    elif tier == "action":
        # Tambah pelanggaran
        violation_count = db.increment_violation(message.author.id, guild_id)

        # Cek eskalasi berdasarkan jumlah pelanggaran kumulatif
        escalation = get_escalated_action(violation_count)

        if escalation and escalation.get("action_override"):
            # Eskalasi override — pelanggaran berulang
            await handle_escalated_action(
                message, confidence, severity, violation_count, escalation
            )
        else:
            # Routing normal berdasarkan severity
            if severity == "mild":
                await handle_mild(message, confidence, severity, violation_count)
            elif severity == "moderate":
                await handle_moderate(message, confidence, severity, violation_count)
            elif severity == "severe":
                await handle_severe(message, confidence, severity, violation_count)
            else:
                # Fallback legacy: tier action tapi severity tidak dikenali
                await handle_mild(message, confidence, severity, violation_count)

        # Log tindakan
        db.log_action(
            user_id=message.author.id,
            guild_id=guild_id,
            action=f"action_{severity}_violation_{violation_count}",
            message_content=raw_text[:500],
            confidence=confidence,
            action_tier=tier,
            severity=severity,
            violation_count=violation_count,
            channel_id=message.channel.id,
            message_id=message.id,
        )

    # ── Catat metrik ──────────────────────────────────────────
    _total_latency_ms = (time.perf_counter() - _msg_start) * 1000
    metrics.record_message(
        total_latency_ms=_total_latency_ms,
        tier=tier,
        used_fallback=used_fallback,
    )

    await bot.process_commands(message)


@bot.event
async def on_close():
    if http_session:
        await http_session.close()


# ──────────────────────────────────────────────────────────────
# Slash Commands
# ──────────────────────────────────────────────────────────────
@tree.command(name="status", description="Lihat status bot dan statistik deteksi server ini")
async def cmd_status(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)   # cegah timeout 3s Discord
    guild_id   = interaction.guild_id
    is_enabled = detection_enabled.get(guild_id, True)
    stats      = db.get_guild_stats(guild_id)
    thresholds = Config.get_severity_thresholds()

    embed = discord.Embed(
        title="🛡️ Cyberbullying Detection — Status",
        color=COLOR_OK if is_enabled else discord.Color.dark_gray(),
    )
    embed.add_field(
        name="🟢 Deteksi Aktif" if is_enabled else "🔴 Deteksi Nonaktif",
        value="Ya" if is_enabled else "Tidak", inline=True,
    )
    embed.add_field(name="📊 Total Deteksi",   value=str(stats["total_detections"]), inline=True)
    embed.add_field(name="👥 User Ditandai",   value=str(stats["unique_users_flagged"]), inline=True)

    embed.add_field(
        name="🎯 Severity Thresholds",
        value=(
            f"Mild    : ≥ {thresholds.get('mild', 0.70)*100:.0f}%\n"
            f"Moderate: ≥ {thresholds.get('moderate', 0.80)*100:.0f}%\n"
            f"Severe  : ≥ {thresholds.get('severe', 0.92)*100:.0f}%"
        ),
        inline=False,
    )

    # Kelompokkan tindakan ke kategori yang lebih bersih agar tidak melebihi batas 1024 karakter embed Discord
    aggregated_actions = {}
    for action_key, count in stats.get("actions", {}).items():
        if action_key == "detected_ignore":
            label = "Pesan Aman (Abaikan)"
        elif action_key == "detected_flag":
            label = "Ditandai untuk Review (Flagged)"
        elif action_key == "detected_action":
            label = "Deteksi Pelanggaran (Action)"
        elif action_key.startswith("action_mild"):
            label = "Tindakan Ringan (Mild Action)"
        elif action_key.startswith("action_moderate"):
            label = "Tindakan Sedang (Moderate Action)"
        elif action_key.startswith("action_severe"):
            label = "Tindakan Berat (Severe Action)"
        elif action_key == "mod_warn":
            label = "Pemberian Peringatan Moderator"
        elif action_key == "mod_timeout_30m":
            label = "Timeout 30m oleh Moderator"
        else:
            label = action_key
        
        aggregated_actions[label] = aggregated_actions.get(label, 0) + count

    action_text = "\n".join(f"  `{k}`: {v}" for k, v in aggregated_actions.items())
    if action_text:
        # Batasi panjang value embed maksimal 1024 karakter untuk mencegah crash
        if len(action_text) > 1000:
            action_text = action_text[:997] + "..."
        embed.add_field(name="📋 Tindakan", value=action_text, inline=False)

    embed.set_footer(text=f"Server: {interaction.guild.name}")
    embed.timestamp = datetime.now(timezone.utc)
    await interaction.followup.send(embed=embed, ephemeral=True)


@tree.command(name="violations", description="[MOD] Lihat riwayat pelanggaran user")
@app_commands.describe(user="User yang ingin diperiksa")
async def cmd_violations(interaction: discord.Interaction, user: discord.Member):
    if not interaction.user.guild_permissions.manage_messages:
        await interaction.response.send_message(
            "❌ Kamu tidak punya izin untuk menggunakan perintah ini.", ephemeral=True
        )
        return

    await interaction.response.defer(ephemeral=True)   # cegah timeout 3s Discord
    row  = db.get_violations(user.id, interaction.guild_id)
    logs = db.get_user_logs(user.id, interaction.guild_id, limit=5)

    embed = discord.Embed(
        title=f"📋 Riwayat Pelanggaran — {user.display_name}",
        color=COLOR_INFO,
    )

    if row:
        is_muted, muted_until = db.is_muted(user.id, interaction.guild_id)
        embed.add_field(name="🔢 Total Pelanggaran", value=str(row["violation_count"]), inline=True)
        embed.add_field(name="📅 Terakhir",          value=row["last_violation_at"][:10] if row["last_violation_at"] else "-", inline=True)
        embed.add_field(name="🔇 Status Mute",       value=f"Ya (s/d <t:{int(muted_until.timestamp())}:f>)" if is_muted else "Tidak", inline=True)
    else:
        embed.description = "✅ User ini tidak memiliki riwayat pelanggaran."

    if logs:
        log_text = "\n".join(
            f"• `{l['action']}` | sev={l.get('severity','?')} | {l['timestamp'][:16]} (conf: {l.get('confidence', 0):.2f})"
            for l in logs
        )
        embed.add_field(name="📜 Log Terakhir (5)", value=log_text, inline=False)

    await interaction.followup.send(embed=embed, ephemeral=True)


@tree.command(name="reset_violations", description="[ADMIN] Reset pelanggaran user")
@app_commands.describe(user="User yang akan direset pelanggarannya")
async def cmd_reset(interaction: discord.Interaction, user: discord.Member):
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message(
            "❌ Hanya admin yang bisa menggunakan perintah ini.", ephemeral=True
        )
        return

    db.reset_violations(user.id, interaction.guild_id)

    # Cabut mute role jika ada
    mute_role = interaction.guild.get_role(Config.MUTE_ROLE_ID)
    if mute_role and mute_role in user.roles:
        await user.remove_roles(mute_role, reason=f"Reset oleh {interaction.user}")

    await interaction.response.send_message(
        f"✅ Pelanggaran {user.mention} berhasil direset.", ephemeral=True
    )
    logger.info(f"RESET | target={user} | by={interaction.user}")


@tree.command(name="toggle", description="[ADMIN] Aktifkan/nonaktifkan deteksi cyberbullying")
async def cmd_toggle(interaction: discord.Interaction):
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message(
            "❌ Hanya admin yang bisa menggunakan perintah ini.", ephemeral=True
        )
        return

    guild_id = interaction.guild_id
    current  = detection_enabled.get(guild_id, True)
    detection_enabled[guild_id] = not current

    status = "✅ Diaktifkan" if not current else "🔴 Dinonaktifkan"
    await interaction.response.send_message(
        f"Deteksi cyberbullying sekarang: **{status}**", ephemeral=True
    )
    logger.info(f"TOGGLE | guild={interaction.guild.name} | enabled={not current}")


@tree.command(name="appeal", description="Ajukan banding jika kamu merasa tindakan bot tidak tepat")
@app_commands.describe(alasan="Jelaskan mengapa pesan kamu bukan cyberbullying")
async def cmd_appeal(interaction: discord.Interaction, alasan: str):
    db.add_appeal(
        user_id=interaction.user.id,
        guild_id=interaction.guild_id,
        reason=alasan,
    )

    await interaction.response.send_message(
        "✅ **Banding kamu telah diterima.**\n\n"
        "Moderator akan meninjau permintaanmu dan menghubungi kamu jika diperlukan.\n"
        f"*Alasan yang kamu berikan:* {alasan}",
        ephemeral=True,
    )

    # Notifikasi ke channel moderasi dengan tombol review
    mod_channel = await get_mod_channel(interaction.guild)
    if mod_channel:
        # Ambil appeal ID terbaru user ini
        pending = db.get_pending_appeals(interaction.guild_id, limit=1)
        appeal_id = pending[0]["id"] if pending else 0

        embed = discord.Embed(
            title="📩 Permintaan Banding (Appeal)",
            description=f"{interaction.user.mention} mengajukan banding.",
            color=COLOR_INFO,
        )
        embed.add_field(
            name="👤 User",
            value=f"{interaction.user.mention} (`{interaction.user.id}`)",
            inline=True,
        )
        embed.add_field(name="🆔 Appeal ID", value=f"`#{appeal_id}`", inline=True)
        embed.add_field(name="📝 Alasan",    value=alasan[:500], inline=False)

        # Riwayat pelanggaran singkat
        row = db.get_violations(interaction.user.id, interaction.guild_id)
        if row:
            embed.add_field(
                name="📊 Riwayat Pelanggaran",
                value=f"Total: {row['violation_count']} | Terakhir: {row['last_violation_at'][:10] if row['last_violation_at'] else '-'}",
                inline=False,
            )

        embed.timestamp = datetime.now(timezone.utc)

        view = AppealReviewView(appeal_id=appeal_id, appellant_id=interaction.user.id)
        await mod_channel.send(embed=embed, view=view)


@tree.command(name="mod_review", description="[MOD] Lihat antrian review moderator")
async def cmd_mod_review(interaction: discord.Interaction):
    """Tampilkan antrian mod review yang masih pending."""
    if not interaction.user.guild_permissions.manage_messages:
        await interaction.response.send_message(
            "❌ Kamu tidak punya izin untuk menggunakan perintah ini.", ephemeral=True
        )
        return

    await interaction.response.defer(ephemeral=True)   # cegah timeout 3s Discord
    pending_reviews  = db.get_pending_reviews(interaction.guild_id, limit=10)
    pending_appeals  = db.get_pending_appeals(interaction.guild_id, limit=10)

    embed = discord.Embed(
        title="📋 Antrian Mod Review",
        color=COLOR_INFO,
    )

    if pending_reviews:
        review_lines = []
        for r in pending_reviews[:5]:
            ts   = r["timestamp"][:16]
            sev  = SEVERITY_LABEL.get(r.get("severity", ""), r.get("severity", "?"))
            uid  = r["user_id"]
            member = interaction.guild.get_member(int(uid))
            name = member.display_name if member else f"User#{uid}"
            review_lines.append(
                f"• `#{r['id']}` — **{name}** | {sev} | conf={r.get('confidence', 0):.2f} | {ts}"
            )
        embed.add_field(
            name=f"🔍 Pesan Pending Review ({len(pending_reviews)})",
            value="\n".join(review_lines),
            inline=False,
        )
    else:
        embed.add_field(
            name="🔍 Pesan Pending Review",
            value="✅ Tidak ada review pending.",
            inline=False,
        )

    if pending_appeals:
        appeal_lines = []
        for a in pending_appeals[:5]:
            ts   = a["timestamp"][:16]
            uid  = a["user_id"]
            member = interaction.guild.get_member(int(uid))
            name = member.display_name if member else f"User#{uid}"
            reason_preview = (a.get("reason") or "")[:40]
            appeal_lines.append(
                f"• `#{a['id']}` — **{name}** | _{reason_preview}..._ | {ts}"
            )
        embed.add_field(
            name=f"📩 Appeal Pending ({len(pending_appeals)})",
            value="\n".join(appeal_lines),
            inline=False,
        )
    else:
        embed.add_field(
            name="📩 Appeal Pending",
            value="✅ Tidak ada appeal pending.",
            inline=False,
        )

    embed.set_footer(text=f"Server: {interaction.guild.name}")
    embed.timestamp = datetime.now(timezone.utc)
    await interaction.followup.send(embed=embed, ephemeral=True)


@tree.command(name="metrics", description="[MOD] Lihat metrik performa bot: latency, throughput, reliability")
@app_commands.describe(
    export="(Opsional) Download data sebagai file: csv = spreadsheet, txt = tabel teks, none = hanya embed"
)
@app_commands.choices(export=[
    app_commands.Choice(name="CSV — untuk Excel / Google Sheets", value="csv"),
    app_commands.Choice(name="TXT — tabel teks (mudah dibaca)", value="txt"),
    app_commands.Choice(name="Tidak perlu file", value="none"),
])
async def cmd_metrics(interaction: discord.Interaction, export: str = "none"):
    """Tampilkan metrik evaluasi sistem bot secara real-time."""
    if not (interaction.user.guild_permissions.manage_messages
            or interaction.user.guild_permissions.administrator):
        await interaction.response.send_message(
            "❌ Hanya moderator atau admin yang bisa menggunakan perintah ini.",
            ephemeral=True,
        )
        return

    await interaction.response.defer(ephemeral=True)   # cegah timeout 3s Discord
    snap = metrics.snapshot()

    def _fmt_ms(val) -> str:
        return f"{val:.0f} ms" if val is not None else "-"

    def _fmt_pct(val) -> str:
        return f"{val:.1f}%" if val is not None else "-"

    embed = discord.Embed(
        title="📊 Performance Metrics — Evaluasi Sistem Bot",
        description=(
            f"Snapshot waktu nyata dari **{snap['window_size']}** request terakhir "
            f"(max window: {snap['max_window_size']})."
        ),
        color=COLOR_INFO,
    )

    embed.add_field(
        name="🕐 Latency (End-to-End)",
        value=(
            f"P50 (Median) : `{_fmt_ms(snap['total_p50'])}`\n"
            f"P95          : `{_fmt_ms(snap['total_p95'])}`\n"
            f"P99          : `{_fmt_ms(snap['total_p99'])}`\n"
            f"Rata-rata    : `{_fmt_ms(snap['total_avg'])}`"
        ),
        inline=True,
    )
    embed.add_field(
        name="🔌 Latency Per-Komponen",
        value=(
            f"API P50  : `{_fmt_ms(snap['api_p50'])}`\n"
            f"API P95  : `{_fmt_ms(snap['api_p95'])}`\n"
            f"n8n P50  : `{_fmt_ms(snap['n8n_p50'])}`\n"
            f"n8n P95  : `{_fmt_ms(snap['n8n_p95'])}`"
        ),
        inline=True,
    )

    embed.add_field(name="\u200b", value="\u200b", inline=False)

    embed.add_field(
        name="⚡ Throughput",
        value=(
            f"Pesan diproses/menit  : `{snap['msg_per_minute']}`\n"
            f"Deteksi aktif/menit   : `{snap['det_per_minute']}`\n"
            f"Total pesan diproses  : `{snap['total_messages']:,}`\n"
            f"Total deteksi aktif   : `{snap['total_detections']:,}`"
        ),
        inline=True,
    )

    embed.add_field(
        name="✅ Reliability",
        value=(
            f"API Success Rate  : `{_fmt_pct(snap['api_success_rate'])}`\n"
            f"n8n Success Rate  : `{_fmt_pct(snap['n8n_success_rate'])}`\n"
            f"Fallback Rate     : `{_fmt_pct(snap['fallback_rate'])}`\n"
            f"Error Count       : `{snap['error_count']}`\n"
            f"Uptime            : `{snap['uptime_str']}`"
        ),
        inline=True,
    )

    embed.add_field(
        name="📋 Detail Request",
        value=(
            f"API requests: {snap['api_requests']:,} "
            f"(sukses: {snap['api_successes']:,})\n"
            f"n8n requests: {snap['n8n_requests']:,} "
            f"(sukses: {snap['n8n_successes']:,})\n"
            f"Fallback n8n→API: {snap['fallback_count']:,}x"
        ),
        inline=False,
    )

    embed.set_footer(text=f"Snapshot: {snap['snapshot_at']} | Server: {interaction.guild.name}")
    embed.timestamp = datetime.now(timezone.utc)

    # ── Buat file attachment jika diminta ─────────────────────
    file_attachment = None
    if export in ("csv", "txt"):
        import csv
        import io

        # Ambil riwayat dari database (max 48 entri ≈ 24 jam @ 30 mnt)
        history = db.get_performance_history(limit=48)

        # ── Baris snapshot saat ini (in-memory, belum tersimpan) ──
        now_row = {
            "timestamp":        snap.get("snapshot_at", ""),
            "source":           "live",
            "total_messages":   snap.get("total_messages"),
            "total_detections": snap.get("total_detections"),
            "msg_per_minute":   snap.get("msg_per_minute"),
            "det_per_minute":   snap.get("det_per_minute"),
            "latency_avg_ms":   snap.get("total_avg"),
            "latency_p50_ms":   snap.get("total_p50"),
            "latency_p95_ms":   snap.get("total_p95"),
            "latency_p99_ms":   snap.get("total_p99"),
            "api_p50_ms":       snap.get("api_p50"),
            "api_p95_ms":       snap.get("api_p95"),
            "n8n_p50_ms":       snap.get("n8n_p50"),
            "n8n_p95_ms":       snap.get("n8n_p95"),
            "api_requests":     snap.get("api_requests"),
            "api_successes":    snap.get("api_successes"),
            "api_success_rate": snap.get("api_success_rate"),
            "n8n_requests":     snap.get("n8n_requests"),
            "n8n_successes":    snap.get("n8n_successes"),
            "n8n_success_rate": snap.get("n8n_success_rate"),
            "fallback_count":   snap.get("fallback_count"),
            "fallback_rate":    snap.get("fallback_rate"),
            "error_count":      snap.get("error_count"),
            "uptime_seconds":   snap.get("uptime_seconds"),
            "uptime_str":       snap.get("uptime_str"),
        }

        # ── Gabungkan: snapshot sekarang + riwayat DB ──────────
        # Riwayat DB mungkin sudah punya snapshot yang sama, tapi
        # lebih baik live data selalu muncul paling atas.
        COLUMNS = list(now_row.keys())

        if export == "csv":
            # Peta nama kolom DB → nama kolom output CSV
            DB_TO_CSV = {
                "avg_latency_ms": "latency_avg_ms",
                "p95_latency_ms": "latency_p95_ms",
                "p99_latency_ms": "latency_p99_ms",
            }
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(now_row)
            for row in history:
                row_out = {}
                for col in COLUMNS:
                    # Cari di DB pakai nama DB atau langsung
                    db_col = {v: k for k, v in DB_TO_CSV.items()}.get(col, col)
                    row_out[col] = row.get(db_col) if row.get(db_col) is not None else row.get(col)
                row_out["source"] = "db"
                writer.writerow(row_out)

            filename = f"metrics_{interaction.guild.name}_{snap['snapshot_at'][:10]}.csv"
            file_bytes = buf.getvalue().encode("utf-8-sig")  # utf-8-sig agar Excel baca BOM
            file_attachment = discord.File(
                fp=io.BytesIO(file_bytes),
                filename=filename,
                description="Data evaluasi performa bot (CSV)",
            )

        elif export == "txt":
            # Buat tabel teks berformat rapi
            lines = []
            lines.append("=" * 72)
            lines.append("  LAPORAN EVALUASI PERFORMA SISTEM BOT — CYBERBULLYING DETECTION")
            lines.append(f"  Snapshot: {snap['snapshot_at']}  |  Server: {interaction.guild.name}")
            lines.append("=" * 72)
            lines.append("")

            lines.append("── SNAPSHOT SAAT INI (LIVE) ──────────────────────────────────────────")
            lines.append(f"  {'Metrik':<30} {'Nilai':>15}")
            lines.append(f"  {'-'*30} {'-'*15}")
            pairs = [
                ("Latency Avg",          f"{snap.get('total_avg') or '-'} ms"),
                ("Latency P50 (Median)", f"{snap.get('total_p50') or '-'} ms"),
                ("Latency P95",          f"{snap.get('total_p95') or '-'} ms"),
                ("Latency P99",          f"{snap.get('total_p99') or '-'} ms"),
                ("API Latency P50",      f"{snap.get('api_p50') or '-'} ms"),
                ("API Latency P95",      f"{snap.get('api_p95') or '-'} ms"),
                ("n8n Latency P50",      f"{snap.get('n8n_p50') or '-'} ms"),
                ("n8n Latency P95",      f"{snap.get('n8n_p95') or '-'} ms"),
                ("Throughput Msg/mnt",   f"{snap.get('msg_per_minute')}"),
                ("Throughput Det/mnt",   f"{snap.get('det_per_minute')}"),
                ("Total Pesan",          f"{snap.get('total_messages', 0):,}"),
                ("Total Deteksi",        f"{snap.get('total_detections', 0):,}"),
                ("API Success Rate",     f"{snap.get('api_success_rate') or '-'}%"),
                ("n8n Success Rate",     f"{snap.get('n8n_success_rate') or '-'}%"),
                ("Fallback Rate",        f"{snap.get('fallback_rate') or 0}%"),
                ("Error Count",         f"{snap.get('error_count', 0)}"),
                ("Uptime",              snap.get("uptime_str", "-")),
            ]
            for label, val in pairs:
                lines.append(f"  {label:<30} {val:>15}")

            if history:
                lines.append("")
                lines.append("── RIWAYAT SNAPSHOT (DB) ─────────────────────────────────────────────")
                header = f"  {'Timestamp':<22} {'Avg(ms)':>8} {'P95(ms)':>8} {'P99(ms)':>8} {'Msg':>7} {'Det':>7} {'API%':>6} {'n8n%':>6} {'Err':>5}"
                lines.append(header)
                lines.append(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*5}")
                for r in history:
                    ts = (r.get("timestamp") or "")[:19]
                    lines.append(
                        f"  {ts:<22} "
                        f"{str(round(r.get('avg_latency_ms') or 0)):>8} "
                        f"{str(round(r.get('p95_latency_ms') or 0)):>8} "
                        f"{str(round(r.get('p99_latency_ms') or 0)):>8} "
                        f"{str(r.get('total_messages') or 0):>7} "
                        f"{str(r.get('total_detections') or 0):>7} "
                        f"{str(r.get('api_success_rate') or '-'):>6} "
                        f"{str(r.get('n8n_success_rate') or '-'):>6} "
                        f"{str(r.get('error_count') or 0):>5}"
                    )

            lines.append("")
            lines.append("=" * 72)
            lines.append(f"  Diekspor pada: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
            lines.append("=" * 72)

            filename = f"metrics_{interaction.guild.name}_{snap['snapshot_at'][:10]}.txt"
            file_bytes = "\n".join(lines).encode("utf-8")
            file_attachment = discord.File(
                fp=io.BytesIO(file_bytes),
                filename=filename,
                description="Laporan evaluasi performa bot (TXT)",
            )

    # ── Kirim embed + file (jika ada) ─────────────────────────
    if file_attachment:
        await interaction.followup.send(
            embed=embed,
            file=file_attachment,
            ephemeral=True,
        )
    else:
        await interaction.followup.send(embed=embed, ephemeral=True)

    db.save_performance_snapshot(interaction.guild_id, snap)



# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        Config.validate()
    except EnvironmentError as e:
        logger.error(f"\n{e}\n\nSalin .env.example ke .env dan isi semua variabel.")
        sys.exit(1)

    logger.info("Memulai Cyberbullying Detection Bot ...")
    bot.run(Config.DISCORD_TOKEN, log_handler=None)
