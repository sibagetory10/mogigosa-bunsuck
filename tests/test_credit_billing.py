from __future__ import annotations

import json
import tempfile
import threading
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from src.auth import (
    KST,
    acquire_credit_reservation,
    adjust_user_bonus_credits,
    approve_payment_order,
    batch_update_user_tiers,
    can_access_shared_concept_db,
    can_admin_shared_db,
    can_contribute_shared_db,
    cancel_my_pending_order,
    check_credit_quota,
    check_daily_shared_upload_quota,
    consume_daily_free_ai_quota,
    consume_daily_shared_upload_quota,
    get_bonus_credits,
    get_effective_tier,
    get_kst_month_str,
    get_kst_today_str,
    get_monthly_credit_limit,
    get_total_available_credits,
    get_used_monthly_credits,
    get_user_by_username,
    maybe_sweep_expired_users,
    parse_kst_iso,
    refund_daily_free_ai_quota,
    refund_payment_order,
    register_user,
    reject_payment_order,
    reset_user_monthly_credits,
    settle_credit_reservation,
    submit_payment_order,
    summarize_payment_revenue,
    sync_all_users_state,
)
from src.models import PaymentSettings
from src.storage import StorageManager


class CreditBillingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.storage = StorageManager(
            local_root=root / "exam_db",
            concept_root=root / "concept_db",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_legacy_custom_migration_and_api_key_wipe(self) -> None:
        legacy_payload = [
            {
                "username": "legacy_custom",
                "nickname": "레거시유저",
                "password_hash": "dummy",
                "tier": "CUSTOM",
                "custom_api_key": "AIzaSySecretKeyShouldBeWiped",
                "last_used_date": get_kst_today_str(),
                "last_used_month": get_kst_month_str(),
            },
            {
                "username": "legacy_vip",
                "nickname": "VIP유저",
                "password_hash": "dummy",
                "tier": "VIP",
                "custom_api_key": "",
                "last_used_date": get_kst_today_str(),
                "last_used_month": get_kst_month_str(),
            },
        ]
        self.storage.users_path.write_text(
            json.dumps(legacy_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        u_custom = get_user_by_username(self.storage, "legacy_custom")
        u_vip = get_user_by_username(self.storage, "legacy_vip")
        self.assertIsNotNone(u_custom)
        self.assertIsNotNone(u_vip)
        assert u_custom is not None and u_vip is not None
        self.assertEqual(u_custom.tier, "FREE")
        self.assertEqual(u_custom.custom_api_key, "")
        self.assertEqual(u_vip.tier, "PRO")

        persisted = json.loads(self.storage.users_path.read_text(encoding="utf-8"))
        row_custom = next(r for r in persisted if r["username"] == "legacy_custom")
        self.assertEqual(row_custom["tier"], "FREE")
        self.assertEqual(row_custom["custom_api_key"], "")

    def test_base_then_bonus_deduction_and_monthly_reset_preserves_bonus(self) -> None:
        ok, _, u = register_user(self.storage, "student01", "학생1", "1234")
        self.assertTrue(ok and u is not None)

        cnt, err = adjust_user_bonus_credits(
            self.storage, ["student01"], delta_credits=20, reason="테스트 보너스", actor_username="admin"
        )
        self.assertEqual(cnt, 1)
        self.assertIsNone(err)

        u_now = get_user_by_username(self.storage, "student01")
        assert u_now is not None
        self.assertEqual(get_total_available_credits(u_now), 50)

        # 35C 선예치: 기본 30C 전액 소진 + 보너스 5C 차감 -> 잔여 보너스 15C
        ok_res, res, msg = acquire_credit_reservation(self.storage, "student01", 35, "대량 출제")
        self.assertTrue(ok_res, msg)
        assert res is not None
        settle_credit_reservation(self.storage, res, 35)

        u_after = get_user_by_username(self.storage, "student01")
        assert u_after is not None
        self.assertEqual(get_used_monthly_credits(u_after), 30)
        self.assertEqual(get_bonus_credits(u_after), 15)
        self.assertEqual(get_total_available_credits(u_after), 15)

        # 월간 초기화 후에도 보너스 15C는 유지되어야 함
        r_cnt, r_err = reset_user_monthly_credits(self.storage, ["student01"], actor_username="admin")
        self.assertEqual(r_cnt, 1)
        self.assertIsNone(r_err)

        u_reset = get_user_by_username(self.storage, "student01")
        assert u_reset is not None
        self.assertEqual(get_used_monthly_credits(u_reset), 0)
        self.assertEqual(get_bonus_credits(u_reset), 15)
        self.assertEqual(get_total_available_credits(u_reset), 45)

    def test_partial_and_full_refund_restores_bonus_first_and_writes_ledger(self) -> None:
        register_user(self.storage, "student02", "학생2", "1234")
        adjust_user_bonus_credits(self.storage, ["student02"], 10, "보너스")

        # 기본 20C 먼저 소비해 기본 잔여 10C, 보너스 10C 상태 만들기
        ok1, res1, _ = acquire_credit_reservation(self.storage, "student02", 20, "1차 출제")
        self.assertTrue(ok1 and res1 is not None)
        settle_credit_reservation(self.storage, res1, 20)

        # 16C 선예치 (기본 10C + 보너스 6C) -> 12C만 성공 시 4C 환불은 보너스로 우선 복원
        ok2, res2, _ = acquire_credit_reservation(self.storage, "student02", 16, "2차 출제")
        self.assertTrue(ok2 and res2 is not None)
        settle_credit_reservation(self.storage, res2, 12, detail_note="부분 성공 환불")

        u_mid = get_user_by_username(self.storage, "student02")
        assert u_mid is not None
        self.assertEqual(get_used_monthly_credits(u_mid), 30)
        self.assertEqual(get_bonus_credits(u_mid), 8)
        self.assertEqual(get_total_available_credits(u_mid), 8)

        # 전액 실패(actual=0) 시 전액 환불
        ok3, res3, _ = acquire_credit_reservation(self.storage, "student02", 8, "3차 출제")
        self.assertTrue(ok3 and res3 is not None)
        settle_credit_reservation(self.storage, res3, 0, detail_note="전액 실패 환불")

        u_end = get_user_by_username(self.storage, "student02")
        assert u_end is not None
        self.assertEqual(get_bonus_credits(u_end), 8)
        self.assertEqual(get_total_available_credits(u_end), 8)

        ledger = self.storage.read_credit_ledger(username="student02", limit=20)
        events = [row["event"] for row in ledger]
        self.assertIn("BONUS_GRANT", events)
        self.assertIn("USE", events)
        self.assertIn("REFUND", events)

    def test_concurrent_reservation_blocks_double_spend_and_negative_balance(self) -> None:
        register_user(self.storage, "student03", "학생3", "1234")
        # 20C 사용하여 잔액 10C만 남김
        ok_init, res_init, _ = acquire_credit_reservation(self.storage, "student03", 20, "초기 사용")
        self.assertTrue(ok_init and res_init is not None)
        settle_credit_reservation(self.storage, res_init, 20)

        barrier = threading.Barrier(2)
        results: list[tuple[bool, str]] = []
        lock = threading.Lock()

        def _worker() -> None:
            barrier.wait()
            ok_w, res_w, msg_w = acquire_credit_reservation(self.storage, "student03", 10, "동시 클릭")
            with lock:
                results.append((ok_w, msg_w))
            if ok_w and res_w is not None:
                settle_credit_reservation(self.storage, res_w, 10)

        t1 = threading.Thread(target=_worker)
        t2 = threading.Thread(target=_worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        success_cnt = sum(1 for ok_w, _ in results if ok_w)
        self.assertEqual(success_cnt, 1)
        u_final = get_user_by_username(self.storage, "student03")
        assert u_final is not None
        self.assertEqual(get_total_available_credits(u_final), 0)
        ok_more, avail_more, _, _ = check_credit_quota(self.storage, "student03", 1, "추가 요청")
        self.assertFalse(ok_more)
        self.assertEqual(avail_more, 0)

    def test_shared_db_permissions_and_daily_free_limits(self) -> None:
        register_user(self.storage, "free_u", "무료", "1234")
        register_user(self.storage, "pro_u", "프로", "1234")
        register_user(self.storage, "admin_u", "관리자", "1234")

        users = self.storage.load_users()
        for u in users:
            if u.username == "pro_u":
                u.tier = "PRO"
            elif u.username == "admin_u":
                u.tier = "ADMIN"
        self.storage.save_users(users)

        u_free = get_user_by_username(self.storage, "free_u")
        u_pro = get_user_by_username(self.storage, "pro_u")
        u_admin = get_user_by_username(self.storage, "admin_u")

        # 권한 분리 검증: PRO는 업로드/기여만 가능, 수정·삭제·NCIC는 ADMIN 단독
        self.assertFalse(can_contribute_shared_db(u_free))
        self.assertFalse(can_access_shared_concept_db(u_free))
        self.assertFalse(can_admin_shared_db(u_free))

        self.assertTrue(can_contribute_shared_db(u_pro))
        self.assertTrue(can_access_shared_concept_db(u_pro))
        self.assertFalse(can_admin_shared_db(u_pro))

        self.assertTrue(can_contribute_shared_db(u_admin))
        self.assertTrue(can_admin_shared_db(u_admin))

        # 0C 무료 AI 일일 한도 검증 (FREE=10회)
        for _ in range(10):
            ok_ai, _ = consume_daily_free_ai_quota(self.storage, "free_u", "AI 해설")
            self.assertTrue(ok_ai)
        ok_11th, _ = consume_daily_free_ai_quota(self.storage, "free_u", "AI 해설")
        self.assertFalse(ok_11th)
        refund_daily_free_ai_quota(self.storage, "free_u")
        ok_retry, _ = consume_daily_free_ai_quota(self.storage, "free_u", "AI 해설")
        self.assertTrue(ok_retry)

        # 공용 DB 신규 업로드 일일 한도 검증 (PRO=3회)
        ok_p0, rem_p0, _ = check_daily_shared_upload_quota(self.storage, "pro_u", 1)
        self.assertTrue(ok_p0)
        self.assertEqual(rem_p0, 3)
        consume_daily_shared_upload_quota(self.storage, "pro_u", 3)
        ok_p3, rem_p3, _ = check_daily_shared_upload_quota(self.storage, "pro_u", 1)
        self.assertFalse(ok_p3)
        self.assertEqual(rem_p3, 0)

    def _setup_payment_settings(self) -> None:
        self.storage.save_payment_settings(
            PaymentSettings(
                enabled=True,
                bank_name="카카오뱅크",
                account_number="3333-01-1234567",
                account_holder="홍길동",
            )
        )

    def test_payment_approval_resets_monthly_credits_preserves_bonus_and_counts_revenue(self) -> None:
        self._setup_payment_settings()
        register_user(self.storage, "pay_u1", "결제유저1", "1234")
        adjust_user_bonus_credits(self.storage, ["pay_u1"], 25, "기존 보너스")

        ok_r, res, _ = acquire_credit_reservation(self.storage, "pay_u1", 20, "사전 사용")
        self.assertTrue(ok_r and res is not None)
        settle_credit_reservation(self.storage, res, 20)

        t0 = datetime(2026, 11, 25, 12, 0, 0, tzinfo=KST)
        ok_s, msg_s, ord_obj = submit_payment_order(
            self.storage, "pay_u1", "PRO", "김입금", now_dt=t0
        )
        self.assertTrue(ok_s and ord_obj is not None, msg_s)
        assert ord_obj is not None

        ok_a, msg_a = approve_payment_order(
            self.storage, ord_obj.order_id, admin_username="admin", now_dt=t0
        )
        self.assertTrue(ok_a, msg_a)

        u_after = get_user_by_username(self.storage, "pay_u1")
        assert u_after is not None
        self.assertEqual(u_after.tier, "PRO")
        self.assertEqual(get_used_monthly_credits(u_after), 0)
        self.assertEqual(get_bonus_credits(u_after), 25)
        self.assertEqual(get_total_available_credits(u_after), 625)
        exp_dt = parse_kst_iso(u_after.tier_expires_at)
        self.assertIsNotNone(exp_dt)
        assert exp_dt is not None
        self.assertEqual((exp_dt - t0).days, 30)

        rev = summarize_payment_revenue(self.storage.load_payment_orders(), t0.strftime("%Y-%m"))
        self.assertEqual(rev["month_approved_krw"], 9900)
        self.assertEqual(rev["month_approved_count"], 1)
        ledger = self.storage.read_credit_ledger(username="pay_u1", limit=10)
        self.assertTrue(any(r["event"] == "PAYMENT_APPROVE" for r in ledger))

    def test_duplicate_pending_blocked_and_cancel_or_reject_allows_resubmit(self) -> None:
        self._setup_payment_settings()
        register_user(self.storage, "pay_u2", "결제유저2", "1234")
        t0 = datetime(2026, 11, 25, 12, 0, 0, tzinfo=KST)

        ok1, _, ord1 = submit_payment_order(self.storage, "pay_u2", "PRO", "유저2", now_dt=t0)
        self.assertTrue(ok1 and ord1 is not None)
        assert ord1 is not None

        ok_dup, _, _ = submit_payment_order(self.storage, "pay_u2", "ULTRA", "유저2", now_dt=t0)
        self.assertFalse(ok_dup)

        ok_can, _ = cancel_my_pending_order(self.storage, "pay_u2", ord1.order_id, now_dt=t0)
        self.assertTrue(ok_can)

        ok2, _, ord2 = submit_payment_order(self.storage, "pay_u2", "ULTRA", "유저2", now_dt=t0)
        self.assertTrue(ok2 and ord2 is not None)
        assert ord2 is not None

        ok_rej, _ = reject_payment_order(
            self.storage, ord2.order_id, admin_username="admin", reason="입금 미확인", now_dt=t0
        )
        self.assertTrue(ok_rej)

        ok3, _, ord3 = submit_payment_order(self.storage, "pay_u2", "ULTRA", "유저2", now_dt=t0)
        self.assertTrue(ok3 and ord3 is not None)

    def test_same_tier_extension_stacks_and_cross_tier_resets_30_days(self) -> None:
        self._setup_payment_settings()
        register_user(self.storage, "pay_u3", "결제유저3", "1234")
        t0 = datetime(2026, 11, 1, 10, 0, 0, tzinfo=KST)

        _, _, ord1 = submit_payment_order(self.storage, "pay_u3", "PRO", "유저3", now_dt=t0)
        assert ord1 is not None
        approve_payment_order(self.storage, ord1.order_id, "admin", now_dt=t0)

        # 10일 뒤 동일 등급 PRO 연장 -> 남은 20일 + 30일 = t0 + 60일
        t1 = t0 + timedelta(days=10)
        _, _, ord2 = submit_payment_order(self.storage, "pay_u3", "PRO", "유저3", now_dt=t1)
        assert ord2 is not None
        approve_payment_order(self.storage, ord2.order_id, "admin", now_dt=t1)

        u_ext = get_user_by_username(self.storage, "pay_u3")
        assert u_ext is not None
        exp_ext = parse_kst_iso(u_ext.tier_expires_at)
        assert exp_ext is not None
        self.assertEqual(exp_ext, t0 + timedelta(days=60))

        # 5일 뒤 ULTRA로 등급 변경 -> 승인 시각(t2) + 30일, 한도 2,400C
        t2 = t1 + timedelta(days=5)
        _, _, ord3 = submit_payment_order(self.storage, "pay_u3", "ULTRA", "유저3", now_dt=t2)
        assert ord3 is not None
        approve_payment_order(self.storage, ord3.order_id, "admin", now_dt=t2)

        u_upg = get_user_by_username(self.storage, "pay_u3")
        assert u_upg is not None
        self.assertEqual(u_upg.tier, "ULTRA")
        self.assertEqual(get_total_available_credits(u_upg), 2400)
        exp_upg = parse_kst_iso(u_upg.tier_expires_at)
        assert exp_upg is not None
        self.assertEqual(exp_upg, t2 + timedelta(days=30))

    def test_zero_lag_expiration_and_background_sweep(self) -> None:
        register_user(self.storage, "exp_u", "만료유저", "1234")
        past_exp = (datetime.now(KST) - timedelta(minutes=5)).isoformat(timespec="seconds")
        users = self.storage.load_users()
        for u in users:
            if u.username == "exp_u":
                u.tier = "PRO"
                u.tier_expires_at = past_exp
        self.storage.save_users(users)

        # 디스크 동기화 전 메모리 상의 UserRecord만으로도 즉시 FREE 판정 확인
        raw_u = next(u for u in self.storage.load_users() if u.username == "exp_u")
        self.assertEqual(get_effective_tier(raw_u), "FREE")
        self.assertEqual(get_monthly_credit_limit(raw_u), 30)
        self.assertFalse(can_contribute_shared_db(raw_u))

        # 백그라운드 자동 스윕 호출 시 users.json이 FREE로 강등되고 원장 기록 확인
        swept = maybe_sweep_expired_users(self.storage, force=True)
        self.assertEqual(swept, 1)
        persisted = next(u for u in self.storage.load_users() if u.username == "exp_u")
        self.assertEqual(persisted.tier, "FREE")
        self.assertEqual(persisted.tier_expires_at, "")
        ledger = self.storage.read_credit_ledger(username="exp_u", limit=10)
        self.assertTrue(any(r["event"] == "SUBSCRIPTION_EXPIRED" for r in ledger))

    def test_refund_blocks_free_30c_revival_and_offsets_overconsumed_debt(self) -> None:
        self._setup_payment_settings()
        now_dt = datetime.now(KST)
        # 시나리오 A: 순수 오입금 (FREE 30C 소진 후 PRO 승인 -> 0C 소비 -> 환불 시 무료 30C 부활 차단)
        register_user(self.storage, "sc_a", "오입금자", "1234")
        _, r_a0, _ = acquire_credit_reservation(self.storage, "sc_a", 30, "무료 30C 소진")
        settle_credit_reservation(self.storage, r_a0, 30)

        _, _, ord_a1 = submit_payment_order(self.storage, "sc_a", "PRO", "오입금자", now_dt=now_dt)
        assert ord_a1 is not None
        approve_payment_order(self.storage, ord_a1.order_id, "admin", now_dt=now_dt)
        ok_rf_a, _ = refund_payment_order(self.storage, ord_a1.order_id, "admin", "오입금 환불", now_dt=now_dt)
        self.assertTrue(ok_rf_a)

        u_a = get_user_by_username(self.storage, "sc_a")
        assert u_a is not None
        self.assertEqual(u_a.tier, "FREE")
        self.assertEqual(get_used_monthly_credits(u_a), 30)
        self.assertEqual(get_total_available_credits(u_a), 0)
        self.assertEqual(u_a.refund_credit_debt, 0)

        # 시나리오 B: 승인 후 400C 소비 뒤 환불 -> 재결제 시 400C 부채 상계 & 익월 전환 시 30C 상계
        register_user(self.storage, "sc_b", "과소비환불", "1234")
        _, r_b0, _ = acquire_credit_reservation(self.storage, "sc_b", 30, "무료 30C 소진")
        settle_credit_reservation(self.storage, r_b0, 30)

        _, _, ord_b1 = submit_payment_order(self.storage, "sc_b", "PRO", "과소비환불", now_dt=now_dt)
        assert ord_b1 is not None
        approve_payment_order(self.storage, ord_b1.order_id, "admin", now_dt=now_dt)
        _, r_b1, _ = acquire_credit_reservation(self.storage, "sc_b", 400, "유료 400C 소비")
        settle_credit_reservation(self.storage, r_b1, 400)

        ok_rf_b, _ = refund_payment_order(self.storage, ord_b1.order_id, "admin", "환불 요청", now_dt=now_dt)
        self.assertTrue(ok_rf_b)

        u_b = get_user_by_username(self.storage, "sc_b")
        assert u_b is not None
        self.assertEqual(u_b.tier, "FREE")
        self.assertEqual(get_total_available_credits(u_b), 0)
        self.assertEqual(u_b.refund_credit_debt, 400)

        rev_after_rf = summarize_payment_revenue(self.storage.load_payment_orders(), now_dt.strftime("%Y-%m"))
        self.assertEqual(rev_after_rf["month_approved_krw"], 0)
        self.assertEqual(rev_after_rf["total_refunded_krw"], 19800)

        # 재결제 시 400C 부채가 즉시 상계되어 가용 200C만 남고 부채 0으로 정리
        _, _, ord_b2 = submit_payment_order(self.storage, "sc_b", "PRO", "과소비환불", now_dt=now_dt)
        assert ord_b2 is not None
        approve_payment_order(self.storage, ord_b2.order_id, "admin", now_dt=now_dt)
        u_b2 = get_user_by_username(self.storage, "sc_b")
        assert u_b2 is not None
        self.assertEqual(get_total_available_credits(u_b2), 200)
        self.assertEqual(u_b2.refund_credit_debt, 0)

        # 익월 리셋 시 부채 상계 검증: sc_b2에 100C 부채 부여 후 다음 달로 동기화 -> FREE 30C 자동 차감, 부채 70C 잔류
        register_user(self.storage, "sc_b_next", "익월상계", "1234")
        users = self.storage.load_users()
        for u in users:
            if u.username == "sc_b_next":
                u.refund_credit_debt = 100
        self.storage.save_users(users)
        sync_all_users_state(self.storage, now_dt=now_dt + timedelta(days=35))
        u_next = next(u for u in self.storage.load_users() if u.username == "sc_b_next")
        self.assertEqual(u_next.monthly_used_credits, 30)
        self.assertEqual(u_next.refund_credit_debt, 70)

    def test_duplicate_extension_refund_restores_prior_paid_subscription(self) -> None:
        self._setup_payment_settings()
        now_dt = datetime(2026, 11, 1, 10, 0, 0, tzinfo=KST)
        register_user(self.storage, "sc_c", "중복연장", "1234")

        _, _, ord1 = submit_payment_order(self.storage, "sc_c", "PRO", "중복연장", now_dt=now_dt)
        assert ord1 is not None
        approve_payment_order(self.storage, ord1.order_id, "admin", now_dt=now_dt)
        _, r1, _ = acquire_credit_reservation(self.storage, "sc_c", 250, "기존 사용")
        settle_credit_reservation(self.storage, r1, 250)

        # 같은 달 15일 뒤 2차 중복 연장 승인 후 30C 사용 -> 2차 건만 환불 시 기존 1차 PRO 만료일 + 280C 사용 복원
        t1 = now_dt + timedelta(days=15)
        _, _, ord2 = submit_payment_order(self.storage, "sc_c", "PRO", "중복연장", now_dt=t1)
        assert ord2 is not None
        approve_payment_order(self.storage, ord2.order_id, "admin", now_dt=t1)
        _, r2, _ = acquire_credit_reservation(self.storage, "sc_c", 30, "연장중 사용")
        settle_credit_reservation(self.storage, r2, 30)

        # 1차 주문 먼저 환불 시도는 순서 가드에 의해 차단되어야 함
        ok_wrong_seq, _ = refund_payment_order(self.storage, ord1.order_id, "admin", now_dt=t1)
        self.assertFalse(ok_wrong_seq)

        ok_rf2, _ = refund_payment_order(self.storage, ord2.order_id, "admin", "중복 입금 환불", now_dt=t1)
        self.assertTrue(ok_rf2)

        ord1_saved = next(o for o in self.storage.load_payment_orders() if o.order_id == ord1.order_id)
        u_c = get_user_by_username(self.storage, "sc_c")
        assert u_c is not None
        self.assertEqual(u_c.tier, "PRO")
        self.assertEqual(u_c.tier_expires_at, ord1_saved.granted_expires_at)
        self.assertEqual(u_c.monthly_used_credits, 280)
        self.assertEqual(get_total_available_credits(u_c), 320)

    def test_manual_promotion_unlimited_excluded_from_revenue_and_rename_migrates_orders(self) -> None:
        self._setup_payment_settings()
        register_user(self.storage, "tester01", "테스터", "1234")
        batch_update_user_tiers(self.storage, {"tester01": "PRO"})
        u_t = get_user_by_username(self.storage, "tester01")
        assert u_t is not None
        self.assertEqual(u_t.tier, "PRO")
        self.assertEqual(u_t.tier_expires_at, "")

        rev = summarize_payment_revenue(self.storage.load_payment_orders(), get_kst_month_str())
        self.assertEqual(rev["month_approved_krw"], 0)

        register_user(self.storage, "g_temp123", "구글유저", "1234")
        submit_payment_order(self.storage, "g_temp123", "PRO", "구글유저")
        self.storage.rename_user_data("g_temp123", "real_student01")
        orders = self.storage.load_payment_orders()
        self.assertTrue(any(o.username == "real_student01" for o in orders))
        self.assertFalse(any(o.username == "g_temp123" for o in orders))


if __name__ == "__main__":
    unittest.main()
