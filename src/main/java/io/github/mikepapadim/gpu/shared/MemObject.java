/*
 * MIT License
 *
 * Copyright (c) 2024, Juan Fumero.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
package io.github.mikepapadim.gpu.shared;

import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroBufferInteger;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Taken from {@url https://github.com/jjfumero/corellama2.levelzero}
 * @param segment
 * @param buffer
 */
public record MemObject(MemorySegment segment, LevelZeroBufferInteger buffer) {

    public float get(int index) {
        return segment.getAtIndex(ValueLayout.JAVA_FLOAT, index);
    }

    public void set(int index, float value) {
        segment.setAtIndex(ValueLayout.JAVA_FLOAT, index, value);
    }

}
