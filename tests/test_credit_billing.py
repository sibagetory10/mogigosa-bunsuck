from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from src.auth import (
    acquire_credit_reservation,
    adjust_user_bonus_credits,
    can_access_shared_concept_db,
    can_admin_shared_db,
    can_contribute_shared_db,
    check_credit_quota,
    check_daily_shared_upload_quota,
    consume_daily_free_ai_quota,
    consume_daily_shared_upload_quota,
    get_bonus_credits,
    get_kst_month_str,
    get_kst_today_str,
    get_total_available_credits,
    get_used_monthly_credits,
    get_user_by_username,
    refund_daily_free_ai_quota,
    register_user,
    reset_user_monthly_credits,
    settle_credit_reservation,
)
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


if __name__ == "__main__":
    unittest.main()
