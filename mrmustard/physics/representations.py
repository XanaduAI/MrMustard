# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# TRUNK


# Now we implement the data-dependent methods.
# An advantage of collecting the code in this way is that instead of
# typechecking we try for the best case and catch the exception.
# It's also nice to keep the different versions of the transformation
# in different representations "side by side" (e.g. for learning).

from __future__ import annotations

from mrmustard.abstract.data import GaussianData
from mrmustard.abstract.representation import Representation
from mrmustard.math import Math

math = Math()


class Husimi(Representation):
    def from_charq(self, charQ):
        try:
            return Husimi(
                GaussianData(math.inv(charQ.cov), charQ.mean, charQ.coeff)
            )  # TODO the mean is probably wrong
        except AttributeError:
            print("Fourier transform of charQ.ket/dm/samples")

    def from_wigner(self, wigner):
        try:
            return Husimi(
                GaussianData(
                    math.qp_to_aadag(wigner.cov + math.eye_like(wigner.cov) / 2, axes=(-2, -1)),
                    math.qp_to_aadag(wigner.mean, axes=(-1,)),
                    wigner.coeff,
                )
            )
        except AttributeError:
            print("conv(wigner.dm, exp(|alpha|^2/2))")

    def from_glauber(self, glauber):
        try:
            return Husimi(GaussianData(glauber.cov + math.eye_like(glauber.cov), glauber.mean))
        except AttributeError:
            print("glauber.dm * exp(|alpha|^2)")

    def from_wavefunctionx(self, wavefunctionX):
        try:
            print("wavefunctionX.gaussian to Husimi...")
        except AttributeError:
            print("wavefunctionX.ket to Husimi...")

    def from_stellar(self, stellar):  # what if hilbert vector?
        try:
            math.Xmat(stellar.cov.shape[-1])
            Q = math.inv(math.eye_like(stellar.cov) - math.Xmat @ stellar.cov)
            return Husimi(GaussianData(Q, Q @ math.Xmat @ stellar.mean))
        except AttributeError:
            print("stellar.ket/dm to Husimi...")


# BRANCHES


class Wigner(Representation):
    def from_husimi(self, husimi):
        try:
            return Wigner(GaussianData(husimi.cov - math.eye_like(husimi.cov) / 2, husimi.mean))
        except AttributeError:
            print("husimi.ket * exp(-|alpha|^2/2)")
        except AttributeError:
            print("husimi.dm * exp(-|alpha|^2)")

    def from_charw(self, charw):
        try:
            return Wigner(GaussianData(math.inv(charw.cov), charw.mean))
        except AttributeError:
            print("Fourier transform of charw.ket")
        except AttributeError:
            print("Fourier transform of charw.dm")


class Glauber(Representation):
    def from_husimi(self, husimi):
        try:
            return Glauber(GaussianData(husimi.cov - math.eye_like(husimi.cov), husimi.mean))
        except AttributeError:
            print("husimi.dm * exp(-|alpha|^2)")

    def from_charp(self, charp):
        try:
            return Glauber(GaussianData(math.inv(charp.cov), charp.mean))
        except AttributeError:
            print("Fourier transform of charp.ket")
        except AttributeError:
            print("Fourier transform of charp.dm")


class WavefunctionX(Representation):
    def from_husimi(self, husimi):
        try:
            print("husimi.gaussian to wavefunctionX.gaussian")
        except AttributeError:
            print("husimi.ket to wavefunctionX.ket...")

    def from_wavefunctionp(self, wavefunctionP):
        try:
            return WavefunctionX(GaussianData(math.inv(wavefunctionP.cov), wavefunctionP.mean))
        except AttributeError:
            print("Fourier transform of wavefunctionP.ket")


class Stellar(Representation):
    # TODO: implement polynomial part (stellar rank > 0)
    def from_husimi(self, husimi):
        try:
            X = math.Xmat(husimi.cov.shape[-1])
            Qinv = math.inv(husimi.cov)
            A = X @ (math.eye_like(husimi.cov) - Qinv)
            return Stellar(
                GaussianData(A, X @ Qinv @ husimi.mean)
            )  # TODO: cov must be the inverse of A though
        except AttributeError:
            print("husimi.ket to stellar...")
        except AttributeError:
            print("husimi.dm to sellar...")

    @property
    def A(self):
        return math.inv(self.cov)

    @property
    def B(self):
        return math.inv(self.cov) @ self.mean

    @property
    def C(self):
        return self.coeff * math.exp(-self.mean.T @ math.inv(self.cov) @ self.mean)


# LEAVES


class CharP(Representation):
    def from_glauber(self, glauber):
        try:
            return CharP(GaussianData(math.inv(glauber.cov), glauber.mean))
        except AttributeError:
            print("Fourier transform of glauber.dm")

    def from_husimi(self, husimi):
        return self.from_glauber(Glauber(husimi))


class CharQ(Representation):
    def from_husimi(self, husimi):
        try:
            return CharQ(GaussianData(math.inv(husimi.cov), husimi.mean))
        except AttributeError:
            print("Fourier transform of husimi.ket")
        except AttributeError:
            print("Fourier transform of husimi.dm")


class CharW(Representation):
    def from_wigner(self, wigner):
        try:
            return CharW(GaussianData(math.inv(wigner.cov), wigner.mean))
        except AttributeError:
            print("Fourier transform of wigner.dm")

    def from_husimi(self, husimi):
        return self.from_wigner(Wigner(husimi))


class WavefunctionP(Representation):
    def from_wavefunctionx(self, wavefunctionX):
        try:
            return WavefunctionP(GaussianData(math.inv(wavefunctionX.cov), wavefunctionX.mean))
        except AttributeError:
            print("Fourier transform of wavefunctionX.ket")

    def from_husimi(self, husimi):
        self.from_wavefunctionx(WavefunctionX(husimi))


class Fock(Representation):
    def from_stellar(self, stellar):
        try:
            print("Recurrence relations")
        except AttributeError:
            print("stellar.ket to Fock...")
        except AttributeError:
            print("stellar.dm to Fock...")
