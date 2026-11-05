#!/usr/bin/env python3
"""
GSP-CHECKUP — быстрая проверка, здоров ли GSP-процессор на GPU.

Что это и зачем
---------------
В современных NVIDIA-драйверах управление картой выполняет не только host-CPU,
но и GSP (GPU System Processor) — небольшой процессор на самой карте.
Драйвер общается с ним через RPC. Если этот канал работает нормально, любой
вызов CUDA API (cudaEventQuery, cudaStreamQuery, cudaMemcpyAsync ...) стоит
единицы микросекунд. Если канал деградировал — те же вызовы стоят сотни
микросекунд, а под нагрузкой уходят в секунды, потому что запросы встают в
очередь к GSP.

Именно это выглядит как "медленная карта": обучение и инференс замедляются в
разы, при этом nvidia-smi показывает нормальную температуру, питание и
загрузку, а ошибок в логах может не быть вовсе.

Как проверяем
-------------
Измеряем время одного вызова cudaEventQuery на событии, которое уже завершено.
Вызов ничего не ждёт на GPU: он лишь спрашивает состояние. Значит всё время
сверх нормы — это стоимость пути CPU -> драйвер -> GSP.

Три режима, потому что они дают разные числа:
  cold   — сразу после создания контекста (у некоторых карт 150-200 мкс)
  warm   — после того, как на карте выполнена работа
  loaded — пока карту держат несколько потоков

Запуск
------
    python3 GSP-CHECKUP.py              # если карт несколько — спросит выбор
    python3 GSP-CHECKUP.py --all        # все карты
    python3 GSP-CHECKUP.py --list       # только перечислить
    python3 GSP-CHECKUP.py -d 0 -d 2    # конкретные карты
    python3 GSP-CHECKUP.py --threads 8  # сколько потоков в режиме loaded

Зависимостей нет: только стандартная библиотека Python и libcudart, которая
уже стоит вместе с драйвером. Никаких весов, моделей и загрузок.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import statistics
import sys
import threading
import time

# ---------------------------------------------------------------------------
# Пороги. Подобраны по живым измерениям: исправная A100 даёт ~2 мкс,
# карта с деградировавшим GSP — 150-180 мкс, то есть в 70 раз хуже.
# ---------------------------------------------------------------------------
HEALTHY_US = 30.0        # warm/loaded p50 ниже этого — карта здорова
BAD_US = 100.0           # выше этого — GSP деградировал
SAMPLES = 100            # сколько измерений делаем для медианы


# ---------------------------------------------------------------------------
# Проверка окружения: без этого скрипт не может работать, и лучше сказать
# об этом прямо, чем падать с невнятной ошибкой.
# ---------------------------------------------------------------------------
def _candidate_libs() -> list[str]:
    """Где искать libcudart. Порядок важен: сначала системный линкер."""
    found: list[str] = []
    guess = ctypes.util.find_library("cudart")
    if guess:
        found.append(guess)
    for path in (
        "/usr/local/cuda/lib64/libcudart.so",
        "/usr/local/cuda/lib64/libcudart.so.13",
        "/usr/local/cuda/lib64/libcudart.so.12",
        "/usr/lib/x86_64-linux-gnu/libcudart.so",
        "/usr/lib/wsl/lib/libcudart.so",
    ):
        if os.path.exists(path):
            found.append(path)
    return found


def load_runtime():
    """Загрузить libcudart и объявить сигнатуры. Возвращает объект или None."""
    problems: list[str] = []
    libs = _candidate_libs()
    if not libs:
        problems.append(
            "не найдена библиотека libcudart\n"
            "  Linux: она лежит рядом с драйвером или в CUDA toolkit\n"
            "    проверьте: ls /usr/local/cuda/lib64/libcudart.so* \n"
            "    или установите toolkit: https://developer.nvidia.com/cuda-downloads"
        )
        return None, problems

    lib = None
    for name in libs:
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError as exc:
            problems.append(f"не удалось загрузить {name}: {exc}")

    if lib is None:
        return None, problems

    # Сигнатуры объявляем явно: иначе ctypes передаст int вместо указателя
    # и получим порчу памяти вместо ответа.
    lib.cudaSetDevice.argtypes = [ctypes.c_int]
    lib.cudaSetDevice.restype = ctypes.c_int
    lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.cudaGetDeviceCount.restype = ctypes.c_int
    lib.cudaDeviceSynchronize.restype = ctypes.c_int
    lib.cudaGetErrorString.argtypes = [ctypes.c_int]
    lib.cudaGetErrorString.restype = ctypes.c_char_p

    lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cudaStreamCreate.restype = ctypes.c_int
    lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    lib.cudaStreamDestroy.restype = ctypes.c_int
    lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    lib.cudaStreamSynchronize.restype = ctypes.c_int

    lib.cudaEventCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                             ctypes.c_uint]
    lib.cudaEventCreateWithFlags.restype = ctypes.c_int
    lib.cudaEventDestroy.argtypes = [ctypes.c_void_p]
    lib.cudaEventDestroy.restype = ctypes.c_int
    lib.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cudaEventRecord.restype = ctypes.c_int
    lib.cudaEventQuery.argtypes = [ctypes.c_void_p]
    lib.cudaEventQuery.restype = ctypes.c_int

    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cudaMalloc.restype = ctypes.c_int
    lib.cudaFree.argtypes = [ctypes.c_void_p]
    lib.cudaFree.restype = ctypes.c_int
    lib.cudaMemsetAsync.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                    ctypes.c_size_t, ctypes.c_void_p]
    lib.cudaMemsetAsync.restype = ctypes.c_int

    return lib, problems


def check_environment(lib, problems: list[str]) -> tuple[bool, int]:
    """Проверить, что карты вообще видны. Возвращает (ok, количество карт)."""
    if lib is None:
        print("НЕ МОЖНО ЗАПУСТИТЬ: не хватает компонентов\n")
        for p in problems:
            print(f"  • {p}")
        print("\nУстановите драйвер NVIDIA (он приносит libcudart и устройства).")
        return False, 0

    count = ctypes.c_int(0)
    rc = lib.cudaGetDeviceCount(ctypes.byref(count))
    if rc != 0:
        msg = lib.cudaGetErrorString(rc).decode(errors="replace")
        print("НЕ МОЖНО ЗАПУСТИТЬ: CUDA не видит устройств\n")
        print(f"  • cudaGetDeviceCount вернул: {msg}")
        print("  • проверьте: nvidia-smi, права доступа к /dev/nvidia*,")
        print("    переменную CUDA_VISIBLE_DEVICES")
        return False, 0

    if count.value == 0:
        print("НЕ МОЖНО ЗАПУСТИТЬ: в системе нет доступных GPU\n")
        print("  • nvidia-smi должен показывать карты")
        print("  • в контейнере нужен --gpus all")
        return False, 0

    return True, count.value


# ---------------------------------------------------------------------------
# Имена карт. Через runtime их не взять, поэтому спрашиваем nvidia-smi.
# Если его нет — не страшно, покажем индекс.
# ---------------------------------------------------------------------------
def device_names(count: int) -> list[str]:
    names = [f"GPU {i}" for i in range(count)]
    smi = _find_nvidia_smi()
    if not smi:
        return names
    try:
        import subprocess
        out = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=20)
        for line in out.stdout.splitlines():
            line = line.strip()
            # формат: "GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-...)"
            if not line.startswith("GPU ") or ": " not in line:
                continue
            idx_part, rest = line.split(": ", 1)
            try:
                idx = int(idx_part.replace("GPU", "").strip())
            except ValueError:
                continue
            model = rest.split(" (")[0].strip()
            if 0 <= idx < count and model:
                names[idx] = model
    except Exception:
        pass
    return names


def _find_nvidia_smi() -> str | None:
    for path in ("/usr/bin/nvidia-smi", "/usr/local/nvidia/bin/nvidia-smi"):
        if os.path.exists(path):
            return path
    from shutil import which
    return which("nvidia-smi")


# ---------------------------------------------------------------------------
# Измерения
# ---------------------------------------------------------------------------
class Device:
    """Одна карта: контекст, поток, событие, буфер."""

    def __init__(self, lib, index: int):
        self.lib = lib
        self.index = index
        self.stream = ctypes.c_void_p()
        self.event = ctypes.c_void_p()
        self.buffer = ctypes.c_void_p()

    def open(self) -> None:
        """Создать контекст, поток, событие и один раз дождаться работы.

        Важно: cudaMalloc здесь ВТОРАЯ операция, а не первая. Дело в том, что
        сама cudaMalloc уже поднимает канал до GSP, и если сделать её первой,
        замер cold потеряет смысл: он будет мерить уже прогретый канал.
        Поэтому сначала создаём событие и поток (они не ходят в GSP), снимаем
        cold, и только потом трогаем память.
        """
        lib = self.lib
        _check(lib, lib.cudaSetDevice(self.index), "cudaSetDevice")

        _check(lib, lib.cudaStreamCreate(ctypes.byref(self.stream)), "cudaStreamCreate")
        _check(lib, lib.cudaEventCreateWithFlags(ctypes.byref(self.event), 0),
               "cudaEventCreate")

    def warm_up(self) -> None:
        """Первая настоящая работа на карте: аллокация + memset + запись события."""
        lib = self.lib
        _check(lib, lib.cudaMalloc(ctypes.byref(self.buffer), 1 << 20), "cudaMalloc")
        self._record_and_wait()

    def _record_and_wait(self) -> None:
        """Записать событие на потоке и дождаться, пока работа завершится.

        После этого cudaEventQuery возвращает 'готово' и не ждёт GPU —
        именно это и позволяет измерить чистую стоимость вызова.
        """
        lib = self.lib
        _check(lib, lib.cudaEventRecord(self.event, self.stream), "cudaEventRecord")
        _check(lib, lib.cudaMemsetAsync(self.buffer, 0, 4096, self.stream),
               "cudaMemsetAsync")
        _check(lib, lib.cudaEventRecord(self.event, self.stream), "cudaEventRecord")
        _check(lib, lib.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")

    def sample(self, n: int = SAMPLES) -> dict:
        """Медиана времени cudaEventQuery по n замерам, в микросекундах."""
        lib = self.lib
        query = lib.cudaEventQuery
        ev = self.event
        xs: list[float] = []
        for _ in range(n):
            t0 = time.perf_counter_ns()
            rc = query(ev)
            dt = (time.perf_counter_ns() - t0) / 1000.0
            if rc not in (0, 1):  # 0 = success, 1 = notReady; иное — ошибка
                raise RuntimeError(f"cudaEventQuery вернул {rc}: "
                                   f"{self.errstr(query, ev)}")
            xs.append(dt)
        xs.sort()
        return {
            "p50": xs[len(xs) // 2],
            "p90": xs[int(len(xs) * 0.9)],
            "max": xs[-1],
        }

    def busy_worker(self, stop: threading.Event) -> None:
        """Держать карту занятой, пока не попросят остановиться.

        Ему нужен СВОЙ контекст-поток: cudaSetDevice привязывает устройство к
        потоку, поэтому каждый рабочий поток выбирает карту сам.
        """
        lib = self.lib
        lib.cudaSetDevice(self.index)
        stream = ctypes.c_void_p()
        buf = ctypes.c_void_p()
        if lib.cudaMalloc(ctypes.byref(buf), 1 << 20) != 0:
            return
        if lib.cudaStreamCreate(ctypes.byref(stream)) != 0:
            return
        try:
            while not stop.is_set():
                lib.cudaMemsetAsync(buf, 0, 1 << 20, stream)
                lib.cudaStreamSynchronize(stream)
        finally:
            lib.cudaStreamDestroy(stream)
            lib.cudaFree(buf)

    def close(self) -> None:
        lib = self.lib
        if self.event:
            lib.cudaEventDestroy(self.event)
        if self.stream:
            lib.cudaStreamDestroy(self.stream)
        if self.buffer:
            lib.cudaFree(self.buffer)

    @staticmethod
    def errstr(lib, *_):
        rc = lib.cudaGetLastError() if hasattr(lib, "cudaGetLastError") else -1
        return lib.cudaGetErrorString(rc).decode(errors="replace")


def _check(lib, rc: int, what: str) -> None:
    if rc != 0:
        msg = lib.cudaGetErrorString(rc).decode(errors="replace")
        raise RuntimeError(f"{what} не удался: {rc} ({msg})")


def probe_device(lib, index: int, threads: int) -> dict:
    """Полный прогон одной карты: cold, warm, loaded."""
    dev = Device(lib, index)
    dev.open()
    try:
        # cold снимаем ДО первой памяти: канал до GSP ещё не поднят работой
        cold = dev.sample(n=30)
        dev.warm_up()
        warm = dev.sample()

        stop = threading.Event()
        workers = [threading.Thread(target=dev.busy_worker, args=(stop,), daemon=True)
                   for _ in range(threads)]
        for w in workers:
            w.start()
        time.sleep(1.0)  # дать потокам выйти на режим
        try:
            loaded = dev.sample(n=50)
        finally:
            stop.set()
            for w in workers:
                w.join(timeout=15)
            lib.cudaDeviceSynchronize()

        return {"index": index, "cold": cold, "warm": warm,
                "loaded": loaded, "threads": threads}
    finally:
        dev.close()


def verdict(row: dict) -> tuple[str, str]:
    """(короткий статус, объяснение).

    Смотрим на warm и loaded. warm — чистый канал до GSP без нашей нагрузки.
    loaded — то же под конкуренцией потоков; именно он уходит в секунды,
    когда запросы встают в очередь к GSP.
    """
    warm = row["warm"]["p50"]
    loaded = row["loaded"]["p50"]

    if warm <= HEALTHY_US and loaded <= HEALTHY_US:
        return "OK", "GSP отвечает быстро и под нагрузкой"
    if warm > BAD_US:
        return "BAD", ("каждый вызов CUDA стоит сотни микросекунд — "
                       "похоже на деградацию канала GSP")
    if loaded > BAD_US:
        return "BAD", ("в покое терпимо, но под нагрузкой задержка растёт "
                       "в сотни раз — запросы встают в очередь к GSP")
    if warm > HEALTHY_US or loaded > HEALTHY_US:
        return "WARN", "задержка повышена, но не в разы — стоит перепроверить"


# ---------------------------------------------------------------------------
# Выбор карт
# ---------------------------------------------------------------------------
def choose_devices(count: int, args) -> list[int]:
    if args.devices:
        bad = [d for d in args.devices if not 0 <= d < count]
        if bad:
            print(f"Нет таких карт: {bad} (доступно 0..{count - 1})")
            sys.exit(2)
        return sorted(set(args.devices))
    if args.all:
        return list(range(count))
    if count == 1:
        return [0]

    names = device_names(count)
    print(f"Найдено карт: {count}\n")
    for i in range(count):
        print(f"  [{i}] {names[i]}")
    print("\nЧто проверять?")
    print("  all             — все карты")
    print("  0,2             — список индексов")
    print("  0               — одна карта")
    print("  q               — выйти")
    try:
        answer = input("\n> ").strip().lower()
    except EOFError:
        answer = "all"
    if answer in ("q", "quit", "exit", ""):
        print("Отменено.")
        sys.exit(0)
    if answer == "all":
        return list(range(count))
    picked: list[int] = []
    for part in answer.replace(" ", "").split(","):
        if part.isdigit() and int(part) < count:
            picked.append(int(part))
    if not picked:
        print("Не понял выбор, беру все карты.")
        return list(range(count))
    return sorted(set(picked))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Проверить, здоров ли GSP-процессор на GPU.")
    ap.add_argument("-d", "--device", dest="devices", type=int, action="append",
                    metavar="N", help="индекс карты; можно указать несколько раз")
    ap.add_argument("--all", action="store_true", help="проверить все карты")
    ap.add_argument("--list", action="store_true", help="только перечислить карты")
    ap.add_argument("--threads", type=int, default=4,
                    help="потоков в режиме loaded (по умолчанию 4)")
    args = ap.parse_args()

    lib, problems = load_runtime()
    ok, count = check_environment(lib, problems)
    if not ok:
        return 2

    if args.list:
        names = device_names(count)
        for i in range(count):
            print(f"  [{i}] {names[i]}")
        return 0

    picked = choose_devices(count, args)
    names = device_names(count)

    print(f"\nПроверяю карт: {len(picked)}  (потоков в нагрузке: {args.threads})")
    print("Задержка — медиана одного вызова cudaEventQuery, микросекунды.\n")
    print(f"{'#':>3} {'модель':<28} {'cold':>8} {'warm':>8} {'loaded':>10}  вердикт")
    print("-" * 78)

    rows = []
    for idx in picked:
        try:
            row = probe_device(lib, idx, args.threads)
        except Exception as exc:  # noqa: BLE001
            print(f"{idx:>3} {names[idx]:<28} {'—':>8} {'—':>8} {'—':>10}  "
                  f"ОШИБКА: {exc}")
            continue
        row["name"] = names[idx]
        status, why = verdict(row)
        row["status"] = status
        row["why"] = why
        rows.append(row)
        print(f"{idx:>3} {names[idx]:<28} {row['cold']['p50']:>8.1f} "
              f"{row['warm']['p50']:>8.1f} {row['loaded']['p50']:>10.1f}  {status}")

    print("-" * 78)

    if not rows:
        print("\nНи одну карту проверить не удалось.")
        return 1

    bad = [r for r in rows if r["status"] == "BAD"]
    warn = [r for r in rows if r["status"] == "WARN"]
    good = [r for r in rows if r["status"] == "OK"]

    print()
    if good:
        print(f"Здоровы ({len(good)}): "
              + ", ".join(f"GPU {r['index']}" for r in good))
    if warn:
        print(f"Под вопросом ({len(warn)}): "
              + ", ".join(f"GPU {r['index']}" for r in warn))
    if bad:
        print(f"С деградацией GSP ({len(bad)}): "
              + ", ".join(f"GPU {r['index']}" for r in bad))

    print()
    if bad:
        print("ИТОГ: обнаружены карты, где канал до GSP работает в разы медленнее нормы.")
        print("      На таких картах обучение и инференс будут тормозить,")
        print("      даже если nvidia-smi показывает нормальную загрузку.")
        print("\n      Что делать:")
        print("        1. сравните VBIOS и Board Part Number у здоровых и больных карт:")
        print("           nvidia-smi -i N -q | grep -E 'VBIOS|Board Part'")
        print("        2. если ревизии разные — обновите VBIOS на отстающих")
        print("        3. исключите больные карты из работы до выяснения")
        return 1

    if warn:
        print("ИТОГ: явной деградации нет, но задержка повышена. Перепроверьте")
        print("      на простое: возможно, карту занимает другая задача.")
        return 1

    print("ИТОГ: все проверенные карты здоровы, GSP отвечает быстро.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
