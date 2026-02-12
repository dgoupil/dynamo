/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package defaulting

import (
	"context"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// admissionCtx builds a context carrying an admission request for the given operation.
func admissionCtx(op admissionv1.Operation) context.Context {
	return admission.NewContextWithRequest(context.Background(), admission.Request{
		AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: op,
		},
	})
}

func TestDGDDefaulter_Default(t *testing.T) {
	const testVersion = "0.8.0"

	tests := []struct {
		name            string
		operatorVersion string
		ctx             context.Context
		dgd             *nvidiacomv1alpha1.DynamoGraphDeployment
		wantAnnotation  string
		wantErr         bool
	}{
		{
			name:            "CREATE stamps operator version on new DGD without annotations",
			operatorVersion: testVersion,
			ctx:             admissionCtx(admissionv1.Create),
			dgd: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
			},
			wantAnnotation: testVersion,
		},
		{
			name:            "CREATE stamps operator version on DGD with existing annotations",
			operatorVersion: testVersion,
			ctx:             admissionCtx(admissionv1.Create),
			dgd: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
					Annotations: map[string]string{
						"some-other-annotation": "some-value",
					},
				},
			},
			wantAnnotation: testVersion,
		},
		{
			name:            "CREATE does not overwrite pre-existing origin version",
			operatorVersion: testVersion,
			ctx:             admissionCtx(admissionv1.Create),
			dgd: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
					Annotations: map[string]string{
						consts.KubeAnnotationDynamoOperatorOriginVersion: "0.7.0",
					},
				},
			},
			wantAnnotation: "0.7.0",
		},
		{
			name:            "UPDATE does not stamp annotation",
			operatorVersion: testVersion,
			ctx:             admissionCtx(admissionv1.Update),
			dgd: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
			},
			wantAnnotation: "",
		},
		{
			name:            "UPDATE preserves existing annotation",
			operatorVersion: testVersion,
			ctx:             admissionCtx(admissionv1.Update),
			dgd: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
					Annotations: map[string]string{
						consts.KubeAnnotationDynamoOperatorOriginVersion: "0.7.0",
					},
				},
			},
			wantAnnotation: "0.7.0",
		},
		{
			name:            "DELETE does not stamp annotation",
			operatorVersion: testVersion,
			ctx:             admissionCtx(admissionv1.Delete),
			dgd: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
			},
			wantAnnotation: "",
		},
		{
			name:            "no admission request in context skips defaulting gracefully",
			operatorVersion: testVersion,
			ctx:             context.Background(),
			dgd: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
			},
			wantAnnotation: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defaulter := NewDGDDefaulter(tt.operatorVersion)

			err := defaulter.Default(tt.ctx, tt.dgd)
			if (err != nil) != tt.wantErr {
				t.Errorf("Default() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			got := ""
			if tt.dgd.Annotations != nil {
				got = tt.dgd.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion]
			}

			if got != tt.wantAnnotation {
				t.Errorf("annotation %q = %q, want %q",
					consts.KubeAnnotationDynamoOperatorOriginVersion, got, tt.wantAnnotation)
			}
		})
	}
}

func TestInitExtraPodSpecContainers(t *testing.T) {
	tests := []struct {
		name string
		dgd  *nvidiacomv1alpha1.DynamoGraphDeployment
		// serviceName -> whether Containers should be non-nil after init
		wantNonNil map[string]bool
	}{
		{
			name: "nil Containers becomes empty slice",
			dgd: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"Frontend": {
							ExtraPodSpec: &nvidiacomv1alpha1.ExtraPodSpec{
								PodSpec: &corev1.PodSpec{
									// Containers is nil (zero value)
								},
							},
						},
					},
				},
			},
			wantNonNil: map[string]bool{"Frontend": true},
		},
		{
			name: "already-set Containers is not modified",
			dgd: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"Worker": {
							ExtraPodSpec: &nvidiacomv1alpha1.ExtraPodSpec{
								PodSpec: &corev1.PodSpec{
									Containers: []corev1.Container{{Name: "sidecar"}},
								},
							},
						},
					},
				},
			},
			wantNonNil: map[string]bool{"Worker": true},
		},
		{
			name: "nil ExtraPodSpec is skipped",
			dgd: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"Router": {},
					},
				},
			},
			wantNonNil: map[string]bool{},
		},
		{
			name: "nil PodSpec is skipped",
			dgd: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"Frontend": {
							ExtraPodSpec: &nvidiacomv1alpha1.ExtraPodSpec{
								MainContainer: &corev1.Container{Name: "main"},
								// PodSpec is nil
							},
						},
					},
				},
			},
			wantNonNil: map[string]bool{},
		},
		{
			name: "multiple services mixed",
			dgd: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"Frontend": {
							ExtraPodSpec: &nvidiacomv1alpha1.ExtraPodSpec{
								PodSpec: &corev1.PodSpec{}, // nil Containers
							},
						},
						"Worker": {
							ExtraPodSpec: &nvidiacomv1alpha1.ExtraPodSpec{
								PodSpec: &corev1.PodSpec{
									Containers: []corev1.Container{{Name: "c"}},
								},
							},
						},
						"Router": {}, // nil ExtraPodSpec
					},
				},
			},
			wantNonNil: map[string]bool{"Frontend": true, "Worker": true},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			initExtraPodSpecContainers(tt.dgd)

			for svcName, svc := range tt.dgd.Spec.Services {
				if svc == nil || svc.ExtraPodSpec == nil || svc.ExtraPodSpec.PodSpec == nil {
					if tt.wantNonNil[svcName] {
						t.Errorf("service %q: expected non-nil Containers", svcName)
					}
					continue
				}
				if tt.wantNonNil[svcName] && svc.ExtraPodSpec.PodSpec.Containers == nil {
					t.Errorf("service %q: Containers should be non-nil", svcName)
				}
			}
		})
	}
}
