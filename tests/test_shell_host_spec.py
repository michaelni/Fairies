"""
/*
 * Copyright (C) 2026 Michael Niedermayer
 *
 * This file is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This file is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License version 2 for more details.
 *
 * Additional permission:
 *
 * Michael Niedermayer is permitted to relicense this file, in whole or
 * in part, under any version of the GNU General Public License, the GNU
 * Affero General Public License, or the GNU Lesser General Public License
 * published by the Free Software Foundation.
 *
 * This additional permission is personal to Michael Niedermayer.  It is
 * not transferable and does not grant any other person permission to
 * relicense this file under a different license.
 *
 * This additional permission may be removed from modified copies of this
 * file.  Removal of this additional permission does not affect the
 * licensing of the file under the GNU General Public License version 2.
 */

Tests for podman_host.parse_shell_host spec parsing."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import podman_host


class ParseShellHostTest(unittest.TestCase):
    def test_bare_spec_gets_default_label_and_limits(self) -> None:
        spec = podman_host.parse_shell_host("fairy@192.0.2.4")
        self.assertEqual("x86_64", spec.label)
        self.assertEqual("fairy@192.0.2.4", spec.host.ssh_dest)
        self.assertEqual(podman_host.CONTAINER_CPUS, spec.cpus)
        self.assertEqual(podman_host.CONTAINER_MEMORY, spec.memory)
        self.assertIsNone(spec.gpu)
        self.assertEqual((), spec.devices)

    def test_full_spec_preserves_inner_equals_in_gpu(self) -> None:
        spec = podman_host.parse_shell_host(
            "arm64=fairy@ampere,cpus=12,memory=16g,gpu=nvidia.com/gpu=0")
        self.assertEqual("arm64", spec.label)
        self.assertEqual("fairy@ampere", spec.host.ssh_dest)
        self.assertEqual("12", spec.cpus)
        self.assertEqual("16g", spec.memory)
        self.assertEqual("nvidia.com/gpu=0", spec.gpu)

    def test_device_repeats_and_joins_gpu_in_podman_args(self) -> None:
        spec = podman_host.parse_shell_host(
            "fairy@h,gpu=nvidia.com/gpu=0,device=/dev/snd/controlC10,device=/dev/snd/pcmC10D1c")
        self.assertEqual(("/dev/snd/controlC10", "/dev/snd/pcmC10D1c"), spec.devices)
        self.assertEqual(
            ("--device=nvidia.com/gpu=0", "--device=/dev/snd/controlC10",
             "--device=/dev/snd/pcmC10D1c"),
            spec.podman_device_args)

    def test_unknown_key_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "disk"):
            podman_host.parse_shell_host("fairy@h,disk=1t")

    def test_label_with_ssh_alias(self) -> None:
        spec = podman_host.parse_shell_host("arm64=armbox")
        self.assertEqual("arm64", spec.label)
        self.assertEqual("armbox", spec.host.ssh_dest)

    def test_bare_ssh_alias(self) -> None:
        spec = podman_host.parse_shell_host("armbox")
        self.assertEqual("x86_64", spec.label)
        self.assertEqual("armbox", spec.host.ssh_dest)

    def test_identity_reaches_remote_host(self) -> None:
        spec = podman_host.parse_shell_host("fairy@h", identity="/k/id")
        self.assertEqual("/k/id", spec.host.identity)

    def test_port_reaches_remote_host(self) -> None:
        spec = podman_host.parse_shell_host("fairy@192.0.2.1,port=17022")
        self.assertEqual(17022, spec.host.port)

    def test_default_port_is_none(self) -> None:
        self.assertIsNone(podman_host.parse_shell_host("fairy@h").host.port)

    def test_non_numeric_port_raises(self) -> None:
        with self.assertRaises(ValueError):
            podman_host.parse_shell_host("fairy@h,port=ssh")


if __name__ == "__main__":
    unittest.main()
